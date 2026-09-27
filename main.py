import importlib.util
import json
import os
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import duckdb
import joblib
import numpy as np
import redis
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "end-to-end-nlp-project")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Final Model Predictions", "final_rf_model.pkl")
LLM_DIR = os.path.join(HERE, "LLM-Integration")
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)
import agentic

RAG_DIR = os.path.join(HERE, "RAG")
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

_rag_spec = importlib.util.spec_from_file_location("rag_integ", os.path.join(RAG_DIR, "rag-integ.py"))
rag_integ = importlib.util.module_from_spec(_rag_spec)
_rag_spec.loader.exec_module(rag_integ)

GROQ_MODEL = "openai/gpt-oss-20b"
RAGAS_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
RAG_TOP_K = 5
RAGAS_INTERACTIVE_EVAL = os.environ.get("RAGAS_INTERACTIVE_EVAL") == "1"

SIM_THRESHOLD = 0.85
MAX_CACHE_SIZE = 500
RAGAS_DB_PATH = os.path.join(RAG_DIR, "ragas_scores.duckdb")

class ModelPrediction(BaseModel):
    model_pred: Literal["high", "medium", "low"] = Field(
        description="Priority predicted by the RandomForest model"
    )


class JudgeVerdict(BaseModel):
    priority: Literal["high", "medium", "low", "none"] = Field(
        description="'none' if the model prediction is correct, otherwise the corrected priority"
    )
    reasoning: str = Field(
        description="One-line reasoning (strictly max 25 words) justifying the priority verdict"
    )
    suggested_action: str = Field(
        description="Short suggested action (strictly max 20 words) to handle this customer ticket"
    )

FEATURE_LABELS = {
    "queue": "routing_queue",
    "tag_1": "primary_topic",
    "tag_3": "topic_subcategory",
    "tag_4": "topic_aspect",
    "tag_5": "supplementary_topic",
}

FEATURE_HELP = {
    "queue": "Support department the ticket is routed to.",
    "tag_1": "Broadest headline issue category.",
    "tag_3": "3rd-level topic refinement.",
    "tag_4": "Operational aspect (recovery/troubleshooting).",
    "tag_5": "Most granular, sparse supplemental detail.",
}

PRIORITY_STYLE = {
    "high": ("HIGH", "🔥", "#dc2626"),
    "medium": ("MEDIUM", "⚠️", "#d97706"),
    "low": ("LOW", "🟢", "#059669"),
}

st.set_page_config(
    page_title="Ticket Priority Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    .hero {
        text-align: center;
        padding: 1.25rem 0 0.25rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
    }
    .hero .sub {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0.35rem;
    }
    .badge {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.28rem 0.9rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #e0e7ff;
    }
    .card {
        background: #ffffff;
        border: 1px solid #eef2f7;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .card-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #635bff);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #4f46e5);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
    }
    .priority-big {
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.02em;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        color: #ffffff;
    }
    .confidence {
        text-align: center;
        margin-top: 0.75rem;
        color: #475569;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .proba-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.55rem 0;
    }
    .proba-label {
        width: 5.5rem;
        text-align: right;
        font-weight: 600;
        font-size: 0.85rem;
        color: #334155;
    }
    .proba-track {
        flex: 1;
        height: 0.55rem;
        border-radius: 999px;
        background: #f1f5f9;
        overflow: hidden;
    }
    .proba-fill {
        height: 100%;
        border-radius: 999px;
    }
    .proba-val {
        width: 3.4rem;
        font-weight: 700;
        font-size: 0.85rem;
        color: #0f172a;
    }
    .sidebar-section {
        margin-bottom: 1.1rem;
    }
    .sidebar-section .k {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .sidebar-section .v {
        font-weight: 600;
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #eef2f7;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stSelectbox label,
    .stTextArea label,
    [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .action-box {
        margin-top: 1.1rem;
        padding: 0.95rem 1.1rem;
        border-radius: 14px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #14532d;
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .action-box .lbl {
        font-weight: 700;
        color: #15803d;
        white-space: nowrap;
    }
    .reason-box {
        margin-top: 1.1rem;
        padding: 0.95rem 1.1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .reason-box .lbl {
        font-weight: 700;
        color: #475569;
        white-space: nowrap;
    }
    .agent-box {
        margin-top: 1.1rem;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fef9ff;
        border: 1px solid #f3e8ff;
        color: #581c87;
        font-size: 0.93rem;
        line-height: 1.5;
    }
    .agent-box .cache-line {
        color: #9333ea;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    .note-box {
        margin-top: 0.8rem;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #78350f;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .learn-box {
        margin-top: 0.8rem;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .learn-box.warn {
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    .foot {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def predict_priority(body: str, cat_values: list[str]) -> tuple:
    arts = load_model()
    classifier = arts["classifier"]
    le = arts["label_encoder"]
    ohe = arts["onehot_encoder"]
    sentence_encoder = arts["sentence_encoder"]

    emb = sentence_encoder.encode([clean_text(body)]).astype(np.float32)
    cat_enc = ohe.transform(np.array([cat_values]))
    X = np.hstack([emb, cat_enc])

    proba = classifier.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    model_pred = ModelPrediction(model_pred=le.classes_[idx])
    return model_pred, proba


def embed_ticket(body: str) -> np.ndarray:
    arts = load_model()
    vec = arts["sentence_encoder"].encode([clean_text(body)])[0]
    return np.asarray(vec, dtype="float32")


@st.cache_resource(show_spinner="Loading RAG index…")
def get_rag_vectorstore():
    return rag_integ.get_vectorstore()


@st.cache_resource(show_spinner="Setting up RAG generator…")
def get_rag_chain():
    return rag_integ.build_rag_chain()


def retrieve_contexts(ticket_body: str, k: int = RAG_TOP_K) -> tuple[list[str], str]:
    results = rag_integ.retrieve_similar_answers(
        ticket_body, k=k, vectorstore=get_rag_vectorstore()
    )
    contexts = [doc.page_content for doc, _ in results]
    context_str = "\n\n".join(
        f"[{rank}] (similarity={score:.4f})\n{doc.page_content}"
        for rank, (doc, score) in enumerate(results, 1)
    )
    return contexts, context_str


def generate_rag_answer(ticket_body: str) -> tuple[str, list[str]]:
    contexts, context_str = retrieve_contexts(ticket_body)
    response = get_rag_chain().invoke(
        {"ticket_body": ticket_body, "context": context_str}
    )
    return response.content, contexts


def _cos_sim(a, b) -> float:
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


class RedisAnswerCache:
    _IDS, _SEQ = "tc:ids", "tc:seq"
    _KF = "tc:{tid}"

    def __init__(self):
        self.r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    def lookup(self, vector) -> tuple[float | None, str | None]:
        best_tid, best_sim, best = None, -1.0, None
        for tid in self.r.smembers(self._IDS):
            h = self.r.hgetall(self._KF.format(tid=tid))
            if not h:
                continue
            sim = _cos_sim(vector, h.get("vector", "[]"))
            if sim > best_sim:
                best_tid, best_sim, best = tid, sim, h
        if best_tid is not None and best_sim >= SIM_THRESHOLD:
            self.r.hincrby(self._KF.format(tid=best_tid), "count", 1)
            return best_sim, best["answer"]
        return None, None

    def put(self, query: str, vector, answer: str) -> None:
        tid = self.r.incr(self._SEQ)
        self.r.hset(
            self._KF.format(tid=tid),
            mapping={
                "query": query,
                "vector": json.dumps(np.asarray(vector, dtype="float32").tolist()),
                "answer": answer,
                "count": 1,
            },
        )
        self.r.sadd(self._IDS, tid)
        if int(self.r.scard(self._IDS)) > MAX_CACHE_SIZE:
            self._evict()

    def _evict(self) -> None:
        victim = min(
            self.r.smembers(self._IDS),
            key=lambda tid: (
                int(self.r.hget(self._KF.format(tid=tid), "count") or 0),
                int(tid),
            ),
        )
        self.r.delete(self._KF.format(tid=victim))
        self.r.srem(self._IDS, victim)


class DictAnswerCache:
    def __init__(self):
        self.entries: dict[int, dict] = {}
        self.seq = 0

    def lookup(self, vector) -> tuple[float | None, str | None]:
        best_sim, best_key = -1.0, None
        for key, entry in self.entries.items():
            sim = _cos_sim(vector, entry["vector"])
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= SIM_THRESHOLD:
            self.entries[best_key]["count"] += 1
            return best_sim, self.entries[best_key]["answer"]
        return None, None

    def put(self, query: str, vector, answer: str) -> None:
        self.seq += 1
        self.entries[self.seq] = {
            "query": query,
            "vector": np.asarray(vector, dtype="float32").tolist(),
            "answer": answer,
            "count": 1,
        }
        if len(self.entries) > MAX_CACHE_SIZE:
            victim = min(self.entries, key=lambda k: (self.entries[k]["count"], k))
            del self.entries[victim]


@st.cache_resource(show_spinner="Connecting to answer cache…")
def get_answer_cache():
    try:
        probe = redis.Redis(host="localhost", port=6379, decode_responses=True)
        if probe.ping():
            probe.close()
            return RedisAnswerCache()
    except Exception:
        pass
    return DictAnswerCache()


@st.cache_resource(show_spinner="Setting up RAG evaluation judge…")
def get_ragas_judge():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=NVIDIA_BASE,
    )


def evaluate_and_score_rag(
    ticket_body: str, contexts: list[str], answer: str, ground_truth: str | None = None
) -> dict:
    import json

    client = get_ragas_judge()

    context_block = "\n\n".join(
        f"[{i}]\n{chunk}"
        for i, chunk in enumerate(
            [c for c in contexts if c and c.strip()], 1
        )
    )

    ground_truth_line = (
        f"\n\nReference (ground truth):\n{ground_truth}" if ground_truth else ""
    )

    judge_prompt = f"""You are an expert RAG evaluation judge. Judge the candidate answer against the retrieved context chunks.

Question: {ticket_body}

Candidate answer:
{answer}

Retrieved context chunks (retrieved answers):
{context_block}
{ground_truth_line}

Score the answer on these four metrics, each as a float in [0.0, 1.0]:
- faithfulness: fraction of claims in the candidate answer that are directly supported by the retrieved context.
- groundness: degree to which the candidate answer is grounded in the retrieved context rather than external knowledge.
- precision: fraction of the retrieved context that is relevant to the question and actually used by the answer.
- recall: fraction of the reference answer's facts that are covered by the retrieved context. Set to null when no reference is provided.

Return only a JSON object with the numeric fields, no keys other than these:
{{"faithfulness": <float>, "groundness": <float>, "precision": <float>, "recall": <float|null>}}"""

    resp = client.chat.completions.create(
        model=RAGAS_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    def _clamp(key: str):
        val = data.get(key)
        if val in (None, "null"):
            return None
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return None

    scores = {
        "faithfulness": _clamp("faithfulness"),
        "groundness": _clamp("groundness"),
        "precision": _clamp("precision"),
        "recall": _clamp("recall") if ground_truth else None,
        "chunks_used": len(contexts),
    }
    return scores


def _connect_ragas_db():
    return duckdb.connect(RAGAS_DB_PATH)


def init_ragas_db() -> None:
    with _connect_ragas_db() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS ragas_scores ("
            "timestamp TIMESTAMP DEFAULT now(), text_body TEXT, precision DOUBLE, "
            "recall DOUBLE, faithfulness DOUBLE, groundness DOUBLE, "
            "chunks_used INTEGER, agent_answer TEXT)"
        )


def log_ragas_score(text_body: str, scores: dict, agent_answer: str) -> str | None:
    try:
        with _connect_ragas_db() as con:
            con.execute(
                "INSERT INTO ragas_scores "
                "(text_body, precision, recall, faithfulness, groundness, "
                "chunks_used, agent_answer) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    text_body,
                    scores.get("precision"),
                    scores.get("recall"),
                    scores.get("faithfulness"),
                    scores.get("groundness"),
                    scores.get("chunks_used"),
                    agent_answer,
                ),
            )
        return None
    except Exception as e:
        return f"eval-store warning: {e}"


def offline_ragas_pass() -> None:
    import ast

    init_ragas_db()
    with open(os.path.join(RAG_DIR, "evals.txt"), encoding="utf-8") as fh:
        raw = fh.read().split("=", 1)[1].strip()
    samples = ast.literal_eval(raw)["samples"]

    print("Offline RAGAS pass over RAG/evals.txt — NVIDIA nemotron-3.5-lightning as judge")
    print(f"{'#':>2}  {'faith':>6} {'ground':>6} {'prec':>6} {'recall':>6} {'chunks':>6}")
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        answer, contexts = generate_rag_answer(question)
        scores = evaluate_and_score_rag(
            question, contexts, answer, ground_truth=sample.get("expected_answer")
        )
        warning = log_ragas_score(question, scores, answer)
        print(
            f"{i:>2}  {scores['faithfulness']!s:>6} {scores['groundness']!s:>6} "
            f"{scores['precision']!s:>6} {scores['recall']!s:>6} {scores['chunks_used']!s:>6}"
        )
        if warning:
            print(f"   {warning}")
    print("Offline pass complete — rows written to RAG/ragas_scores.duckdb")


if os.environ.get("RAG_OFFLINE_EVAL") == "1":
    offline_ragas_pass()
    raise SystemExit(0)


@st.cache_resource(show_spinner="Setting up judge…")
def make_chain():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are the judge LLM in an end-to-end NLP ticket priority classification pipeline.\n"
                    "You receive the ticket body, the model's predicted priority, and the feature schema "
                    "used by the model. Decide whether the model prediction is correct for this ticket.\n"
                    "Return exactly {{\"priority\": \"none\"}} if the model prediction is correct.\n"
                    "If the model prediction is WRONG, return the corrected priority as one of "
                    "{{\"priority\": \"high\"}}, {{\"priority\": \"medium\"}}, {{\"priority\": \"low\"}}.\n"
                    "Your verdict overrides the model, so be decisive and reason about the ticket's severity.\n\n"
                    "Priority classification guidelines:\n"
                    "HIGH PRIORITY\n"
                    "- Business Impact: Critical service down, severe outage, or major revenue-blocking issue.\n"
                    "- Security & Risk: Active security breach, data leak, or severe compliance/legal exposure.\n"
                    "- User Reach: Affects all users, core business operations, or high-tier enterprise clients.\n"
                    "- Workaround: No operational workaround available; system unusable.\n\n"
                    "MEDIUM PRIORITY\n"
                    "- Business Impact: Partial feature failure, degraded performance, or localized service disruption.\n"
                    "- Security & Risk: Minor bug or non-critical vulnerability without immediate exposure risk.\n"
                    "- User Reach: Affects a subset of users or non-critical internal teams.\n"
                    "- Workaround: Temporary workaround exists, but workflow remains inconvenient.\n\n"
                    "LOW PRIORITY\n"
                    "- Business Impact: Minor cosmetic flaw, typo, general inquiry, or feature request.\n"
                    "- Security & Risk: Zero security or financial risk.\n"
                    "- User Reach: Single user issue or routine administrative inquiry.\n"
                    "- Workaround: Full workaround available or no operational impact.\n\n"
                    "Apply these rubrics to the ticket body to assess whether the model's predicted "
                    "priority matches the true severity.\n\n"
                    "ALSO provide a one-line reasoning (strictly max 25 words) for your verdict, e.g. "
                    "'all users blocked with no workaround' for high, or 'cosmetic single-user inquiry' "
                    "for low.\n\n"
                    "ALSO provide a short suggested action for handling this customer ticket "
                    "(strictly within 20 words, e.g. 'forward immediately to the customer support team' "
                    "for high, 'raise a JIRA ticket and assign to the engineering team' for medium)."
                ),
            ),
            (
                "human",
                (
                    "Ticket body:\n{ticket_body}\n\n"
                    "Model predicted priority: {model_pred}\n\n"
                    "Feature schema (suggested names):\n{feature_labels}"
                ),
            ),
        ]
    )

    return prompt | llm.with_structured_output(JudgeVerdict)


@st.cache_resource(show_spinner="Setting up learning agent…")
def get_agent():
    return agentic.build_agent()


st.markdown(
    """
    <div class="hero">
        <h1>🎫 Ticket Priority Classifier</h1>
        <div class="sub">Random Forest prediction, audited and corrected by an <b>LLM judge</b>.</div>
        <span class="badge">Random Forest · 200 trees · LLM-as-Judge pipeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)

arts = load_model()
cat_cols = arts["cat_cols"]
ohe = arts["onehot_encoder"]
init_ragas_db()

with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown(
        """
        <div class="sidebar-section">
            <div class="k">Model</div>
            <div class="v">RandomForestClassifier</div>
        </div>
        <div class="sidebar-section">
            <div class="k">Judge</div>
            <div class="v">{model_name}</div>
        </div>
        <div class="sidebar-section">
            <div class="k">Feature space</div>
            <div class="v">Body embedding (384) + one-hot categories</div>
        </div>
        """.format(model_name=GROQ_MODEL),
        unsafe_allow_html=True,
    )
    st.markdown("### 🧩 Features used")
    for col, label in FEATURE_LABELS.items():
        st.markdown(f"- **{label}**")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.form("ticket_form", clear_on_submit=False):
        st.markdown(
            '<div class="card"><div class="card-title">✍️ Ticket Details</div>',
            unsafe_allow_html=True,
        )

        body = st.text_area(
            "Ticket body",
            height=180,
            placeholder="Paste the customer request / problem description here…",
            help="The message text — this is embedded with a sentence-transformer.",
        )

        options_by_col = {}
        default_idx = {}
        for i, col in enumerate(cat_cols):
            opts = list(ohe.categories_[i])
            options_by_col[col] = opts
            pref = "Other" if "Other" in opts else opts[0]
            default_idx[col] = opts.index(pref)

        c1, c2 = st.columns(2)
        with c1:
            q = st.selectbox(
                FEATURE_LABELS["queue"],
                options=options_by_col["queue"],
                index=default_idx["queue"],
                help=FEATURE_HELP["queue"],
            )
            t1 = st.selectbox(
                FEATURE_LABELS["tag_1"],
                options=options_by_col["tag_1"],
                index=default_idx["tag_1"],
                help=FEATURE_HELP["tag_1"],
            )
            t3 = st.selectbox(
                FEATURE_LABELS["tag_3"],
                options=options_by_col["tag_3"],
                index=default_idx["tag_3"],
                help=FEATURE_HELP["tag_3"],
            )
        with c2:
            t4 = st.selectbox(
                FEATURE_LABELS["tag_4"],
                options=options_by_col["tag_4"],
                index=default_idx["tag_4"],
                help=FEATURE_HELP["tag_4"],
            )
            t5 = st.selectbox(
                FEATURE_LABELS["tag_5"],
                options=options_by_col["tag_5"],
                index=default_idx["tag_5"],
                help=FEATURE_HELP["tag_5"],
            )
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Priority")

        if submitted:
            if not clean_text(body):
                st.warning("Please enter a ticket body before predicting.")
            else:
                pipeline_started = time.perf_counter()
                cat_values = [q, t1, t3, t4, t5]
                try:
                    with st.spinner("Predicting priority and preparing the response…"):
                        model_pred_json, _ = predict_priority(body, cat_values)
                    query_clean = clean_text(body)
                    vector = embed_ticket(body)
                    cache = get_answer_cache()
                    agent_answer, cache_msg, scores, eval_warning = None, None, None, None

                    cache_sim, cached_ans = cache.lookup(vector)

                    judge_input = {
                        "ticket_body": query_clean,
                        "model_pred": model_pred_json.model_pred,
                        "feature_labels": FEATURE_LABELS,
                    }
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        judge_future = executor.submit(lambda: make_chain().invoke(judge_input))
                        rag_future = None
                        if cached_ans is None:
                            rag_future = executor.submit(generate_rag_answer, query_clean)

                        verdict = judge_future.result()
                        if rag_future is not None:
                            with st.spinner("Drafting agent answer from top-5 retrieved answers…"):
                                agent_answer, contexts = rag_future.result()

                    llm_priority = verdict.priority
                    llm_agrees = llm_priority == "none" or llm_priority == model_pred_json.model_pred

                    stored = None
                    if not llm_agrees:
                        ticket = {
                            "body": query_clean,
                            "routing_queue": q,
                            "primary_topic": t1,
                            "topic_subcategory": t3,
                            "topic_aspect": t4,
                            "supplementary_topic": t5,
                            "target": llm_priority,
                        }
                        try:
                            with st.spinner("Logging disagreement for retraining…"):
                                stored = agentic.record_disagreement(ticket, agent=get_agent())
                        except Exception as e:
                            stored = f"error: {e}"

                    if cached_ans is not None:
                        agent_answer = cached_ans
                        cache_msg = f"⚡ served from cache (sim {cache_sim:.2f}) — no generation call"
                    else:
                        cache.put(query_clean, vector, agent_answer)
                        if RAGAS_INTERACTIVE_EVAL:
                            try:
                                with st.spinner("Scoring RAG generation with RAGAS (Gemini as judge)…"):
                                    scores = evaluate_and_score_rag(query_clean, contexts, agent_answer)
                                eval_warning = log_ragas_score(query_clean, scores, agent_answer)
                            except Exception as e:
                                eval_warning = f"eval skipped: {e}"

                    st.session_state["model_pred"] = model_pred_json.model_pred
                    st.session_state["llm_priority"] = llm_priority
                    st.session_state["llm_agrees"] = llm_agrees
                    st.session_state["reasoning"] = verdict.reasoning
                    st.session_state["suggested_action"] = verdict.suggested_action
                    st.session_state["agent_answer"] = agent_answer
                    st.session_state["cache_msg"] = cache_msg
                    st.session_state["ragas_scores"] = scores
                    st.session_state["eval_warning"] = eval_warning
                    st.session_state["cat_values"] = cat_values
                    st.session_state["stored"] = stored
                    st.session_state["latency_seconds"] = time.perf_counter() - pipeline_started
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        '<div class="card"><div class="card-title">📊 Prediction</div>',
        unsafe_allow_html=True,
    )
    if "model_pred" not in st.session_state:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
                <div style="font-size:2.6rem; margin-bottom:0.6rem;">🎯</div>
                <div>Fill in the ticket details and click<br><b>Predict Priority</b> to see the result.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        model_pred = st.session_state["model_pred"]
        llm_priority = st.session_state["llm_priority"]
        action = st.session_state["suggested_action"]
        llm_agrees = st.session_state["llm_agrees"]
        final_priority = model_pred if llm_agrees else llm_priority
        label, icon, color = PRIORITY_STYLE[final_priority]
        st.markdown(
            f'<div class="priority-big" style="background:linear-gradient(135deg,{color},#991b1b);">{icon} {label}</div>',
            unsafe_allow_html=True,
        )
        source = "LLM judge confirmed the model" if llm_agrees else "LLM judge overrode the model"
        st.markdown(
            f'<div class="confidence">{source}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Measured request latency: {st.session_state.get('latency_seconds', 0):.1f}s"
        )

        if not llm_agrees:
            st.markdown(
                f'<div class="note-box">⚠️ Model predicted <b>{model_pred}</b> but the LLM judge '
                f'decided <b>{llm_priority}</b> — the judge&apos;s verdict prevails.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="reason-box"><span class="lbl">🧠 Reasoning:</span> '
            f'{st.session_state.get("reasoning", "")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="action-box"><span class="lbl">🚀 Suggested action:</span> {action}</div>',
            unsafe_allow_html=True,
        )

        agent_answer = st.session_state.get("agent_answer")
        if agent_answer:
            st.markdown("### 🤖 Agent answer (RAG)")
            cache_msg = st.session_state.get("cache_msg")
            cache_line = f'<div class="cache-line">{cache_msg}</div>' if cache_msg else ""
            st.markdown(
                f'<div class="agent-box">{agent_answer}{cache_line}</div>',
                unsafe_allow_html=True,
            )

        scores = st.session_state.get("ragas_scores")
        if scores and scores.get("error"):
            st.markdown(
                f'<div class="learn-box warn">⚠️ RAGAS scoring failed: <code>{scores["error"]}</code></div>',
                unsafe_allow_html=True,
            )
        elif scores:
            labels = {
                "faithfulness": "Faithfulness",
                "groundness": "Groundness",
                "precision": "Precision",
                "recall": "Recall",
                "chunks_used": "Chunks used",
            }
            cols = st.columns(5)
            for col, (key, label) in zip(cols, labels.items()):
                val = scores.get(key)
                with col:
                    st.metric(label, f"{val:.2f}" if isinstance(val, float) else val)

        eval_warning = st.session_state.get("eval_warning")
        if eval_warning and not scores:
            st.markdown(
                f'<div class="learn-box warn">⚠️ {eval_warning}</div>',
                unsafe_allow_html=True,
            )

        stored = st.session_state.get("stored")
        if not llm_agrees and stored:
            if stored.startswith("error"):
                st.markdown(
                    f'<div class="learn-box warn">⚠️ Learning agent could not store the'
                    f' disagreement: <code>{stored}</code></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="learn-box">📥 Disagreement captured for retraining '
                    f'— {stored}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 🧾 Ticket summary")
        cats = st.session_state["cat_values"]
        for col, val in zip(cat_cols, cats):
            st.markdown(f"**{FEATURE_LABELS[col]}** — <code>{val}</code>",
                        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="foot">Ticket Priority Classifier · Random Forest on sentence embeddings + LLM judge</div>',
    unsafe_allow_html=True,
)