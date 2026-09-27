import os
import re
from typing import Literal

import joblib
import numpy as np
import pandas as pd
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
DATA_PATH = os.path.join(ROOT, "Data", "test_evaluation_dataset - Untitled.csv")

GROQ_MODEL = "openai/gpt-oss-20b"

FEATURE_LABELS = {
    "queue": "routing_queue",
    "tag_1": "primary_topic",
    "tag_3": "topic_subcategory",
    "tag_4": "topic_aspect",
    "tag_5": "supplementary_topic",
}

COL_MAP = {
    "queue": "queue",
    "tag1": "tag_1",
    "tag3": "tag_3",
    "tag4": "tag_4",
    "tag5": "tag_5",
}

CAT_COLS = ["queue", "tag_1", "tag_3", "tag_4", "tag_5"]


class TicketFeatures(BaseModel):
    body: str = Field(description="Ticket body text")
    queue: str = Field(default="Other", description="Support department the ticket is routed to")
    tag1: str = Field(default="unknown", description="Broadest headline issue category")
    tag3: str = Field(default="unknown", description="3rd-level topic refinement")
    tag4: str = Field(default="unknown", description="Operational aspect (recovery/troubleshooting)")
    tag5: str = Field(default="unknown", description="Most granular, sparse supplemental detail")


class JudgeVerdict(BaseModel):
    priority: Literal["high", "medium", "low", "none"] = Field(
        description="'none' if the model prediction is correct, otherwise the corrected priority"
    )


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_artifacts() -> dict:
    return joblib.load(MODEL_PATH)


def preprocess(features: TicketFeatures, arts: dict) -> np.ndarray:
    sentence_encoder = arts["sentence_encoder"]
    ohe = arts["onehot_encoder"]
    cat_cols = arts["cat_cols"]

    emb = sentence_encoder.encode([clean_text(features.body)]).astype(np.float32)
    row = {
        "tag_1": features.tag1,
        "tag_3": features.tag3,
        "tag_4": features.tag4,
        "tag_5": features.tag5,
    }
    cat_values = [str(features.queue)] + [str(row[c]) for c in ["tag_1", "tag_3", "tag_4", "tag_5"]]
    cat_enc = ohe.transform(pd.DataFrame([cat_values], columns=cat_cols))
    return np.hstack([emb, cat_enc])


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
                    "priority matches the true severity.\n"
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


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set. Add it to the environment / .env before running.")

    arts = load_artifacts()
    chain = make_chain()

    df = pd.read_csv(DATA_PATH, keep_default_na=True).fillna("unknown")
    df = df.rename(columns=COL_MAP)
    targets = df["priority"].astype(str).str.strip().str.lower()

    model_agree = 0
    pipeline_correct = 0
    total = len(df)

    for i, row in df.iterrows():
        features = TicketFeatures(
            body=row["body"],
            queue=row["queue"],
            tag1=row["tag_1"],
            tag3=row["tag_3"],
            tag4=row["tag_4"],
            tag5=row["tag_5"],
        )

        X = preprocess(features, arts)
        classifier = arts["classifier"]
        le = arts["label_encoder"]
        model_pred = le.classes_[int(np.argmax(classifier.predict_proba(X)[0]))]

        verdict = chain.invoke(
            {
                "ticket_body": features.body,
                "model_pred": model_pred,
                "feature_labels": FEATURE_LABELS,
            }
        )
        llm_decision = verdict.priority
        final_priority = model_pred if llm_decision == "none" else llm_decision
        actual = targets[i]

        agrees = llm_decision == "none" or llm_decision == model_pred
        correct = final_priority == actual

        model_agree += int(agrees)
        pipeline_correct += int(correct)

        print(
            f"{i + 1:3d}) model={model_pred:6s} llm={llm_decision:6s} "
            f"final={final_priority:6s} actual={actual:6s} "
            f"{'OK' if correct else 'X'} {'(llm agrees)' if agrees else ''}"
        )

    model_acc = model_agree / total * 100
    pipe_acc = pipeline_correct / total * 100

    print()
    print("=" * 60)
    print("LLM-as-Judge Pipeline Results")
    print("=" * 60)
    print(f"Rows evaluated      : {total}")
    print(f"Model / LLM agree   : {model_agree}/{total}")
    print(f"Model accuracy      : {model_acc:.2f}%")
    print(f"Pipeline accuracy   : {pipe_acc:.2f}% (LLM decision prevails vs ground truth)")
    print("=" * 60)


if __name__ == "__main__":
    main()