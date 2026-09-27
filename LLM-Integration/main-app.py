import os
import re
from typing import Literal

import joblib
import numpy as np
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
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "Final Model Predictions", "final_rf_model.pkl")

GROQ_MODEL = "openai/gpt-oss-20b"

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


class ModelPrediction(BaseModel):
    model_pred: Literal["high", "medium", "low"] = Field(
        description="Priority predicted by the RandomForest model"
    )


class JudgeVerdict(BaseModel):
    priority: Literal["high", "medium", "low", "none"] = Field(
        description="'none' if the model prediction is correct, otherwise the corrected priority"
    )
    suggested_action: str = Field(
        description="Short suggested action (strictly max 20 words) to handle this customer ticket"
    )


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
    .feature-chip {
        display: inline-block;
        margin: 0 0.15rem 0.4rem 0;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: #4338ca;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
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
    """Run the RF model and return JSON (pydantic) model prediction + probabilities."""
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
            <div class="k">Test accuracy (model)</div>
            <div class="v">0.7867</div>
        </div>
        <div class="sidebar-section">
            <div class="k">Pipeline accuracy</div>
            <div class="v">0.7778 (offline eval)</div>
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
        for col in cat_cols:
            st.markdown(f'<span class="feature-chip">{FEATURE_LABELS[col]}</span>',
                        unsafe_allow_html=True)
        st.markdown(
            '<div style="height:0.2rem"></div>', unsafe_allow_html=True
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
                cat_values = [q, t1, t3, t4, t5]
                try:
                    with st.spinner("Judge is reviewing the prediction…"):
                        model_pred_json, proba = predict_priority(body, cat_values)
                        verdict = make_chain().invoke(
                            {
                                "ticket_body": clean_text(body),
                                "model_pred": model_pred_json.model_pred,
                                "feature_labels": FEATURE_LABELS,
                            }
                        )
                    st.session_state["model_pred"] = model_pred_json.model_pred
                    st.session_state["proba"] = proba
                    st.session_state["llm_priority"] = verdict.priority
                    st.session_state["suggested_action"] = verdict.suggested_action
                    st.session_state["cat_values"] = cat_values
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        '<div class="card"><div class="card-title">📊 Final Decision</div>',
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
        proba = st.session_state["proba"]
        llm_priority = st.session_state["llm_priority"]
        action = st.session_state["suggested_action"]

        llm_agrees = llm_priority == "none"
        final_priority = model_pred if llm_agrees else llm_priority
        label, icon, color = PRIORITY_STYLE[final_priority]
        st.markdown(
            f'<div class="priority-big" style="background:linear-gradient(135deg,{color},#991b1b);">'
            f'{icon} {label}</div>',
            unsafe_allow_html=True,
        )
        source = "LLM judge confirmed the model" if llm_agrees else "LLM judge overrode the model"
        st.markdown(
            f'<div class="confidence">{source} · final priority {final_priority}</div>',
            unsafe_allow_html=True,
        )

        if not llm_agrees:
            st.markdown(
                f'<div class="note-box">⚠️ Model predicted <b>{model_pred}</b> but the LLM judge '
                f'decided <b>{llm_priority}</b> — the judge&apos;s verdict prevails.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="action-box"><span class="lbl">🚀 Suggested action:</span> {action}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        for c in ["high", "low", "medium"]:
            ci = ["high", "low", "medium"].index(c)
            _, ic, col = PRIORITY_STYLE[c]
            pct = proba[ci] * 100
            st.markdown(
                f"""
                <div class="proba-row">
                    <div class="proba-label">{ic} {c.title()}</div>
                    <div class="proba-track"><div class="proba-fill" style="width:{pct:.1f}%;background:{col}"></div></div>
                    <div class="proba-val">{pct:.1f}%</div>
                </div>
                """,
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