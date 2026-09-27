# End-to-End NLP Project — Complete Project Summary

**Package name:** `end-to-end-nlp-project` (v0.1.0)
**Author:** ashrith S
**Python:** ≥ 3.12 (managed with `uv`)
**Goal:** Classify customer support ticket **priority** (`high` / `medium` / `low`) using classical ML on text embeddings, then audit/correct predictions with an LLM-as-judge, capture disagreements for retraining, and retrieve similar historical agent answers via RAG.

---

## 1. One-line overview

An end-to-end multilingual customer-support ticket priority system: **EDA → preprocessing → Random Forest on MiniLM embeddings → MLflow experiment tracking → Streamlit UI → Groq LLM judge → DuckDB feedback agent → FAISS/Gemini RAG for historical answers**.

---

## 2. Problem statement

Given a support ticket (body text + routing metadata/tags), predict its **priority** so tickets can be triaged correctly.

| Aspect | Detail |
|--------|--------|
| Target | `priority` ∈ {`high`, `medium`, `low`} |
| Primary text feature | Ticket `body` (embedded) |
| Categorical features | `queue`, `tag_1`, `tag_3`, `tag_4`, `tag_5` |
| Languages in dataset | English (~57%) and German (~43%) |
| Modelling language used | **English only** for the trained RF pipeline |

---

## 3. Repository layout

```
End To End NLP Project/
├── main.py                          # Primary Streamlit app (RF + LLM judge + learning agent)
├── test.py                          # Batch CLI: sample tickets → RF + judge + agentic store
├── pyproject.toml / uv.lock         # Dependencies & package metadata
├── .env                             # API keys (GROQ, GEMINI, LangSmith, etc.) — not for commit
├── README.md                        # Placeholder (empty)
│
├── Data/
│   ├── aa_dataset-tickets-multi-lang-5-2-50-version.csv   # Main dataset (~28,587 rows)
│   ├── sample_test_dataset.csv                            # ~200 sample tickets for CLI tests
│   ├── test_evaluation_dataset - Untitled.csv             # Small labelled eval set (~99 rows)
│   └── test.py                                            # RF-only accuracy check on eval set
│
├── Data Analysis /
│   └── data-analysis.ipynb          # Read-only EDA notebook
│
├── Opencode/
│   ├── analysis.md                  # Full column analysis write-up
│   ├── thinking.md                  # Agent working-style / phasing rules
│   └── project.md                   # This file
│
├── Pre Processing + Model Eval/
│   ├── preprocessing.py             # Clean → MiniLM embed → OHE → train/test split
│   ├── evaluation.py                # Model compare, GridSearch, ensemble, final RF + MLflow
│   ├── final_rf_confusion_matrix.png
│   ├── mlflow.db / mlruns/          # Local MLflow tracking
│   └── run_out.log                  # Last training run metrics
│
├── Final Model Predictions/
│   ├── final.py                     # Retrain RF with best params & export joblib bundle
│   └── final_rf_model.pkl           # Deployed artifact (~161 MB)
│
├── LLM-Integration/
│   ├── llm-integ.py                 # Batch LLM-as-judge evaluation vs ground truth
│   ├── main-app.py                  # Earlier Streamlit UI (judge only, no agent)
│   ├── agentic.py                   # LangGraph ReAct agent → DuckDB disagreement store
│   └── feedback_tickets.duckdb      # Feedback DB for RF vs LLM disagreements
│
├── RAG/
│   ├── rag-integ.py                 # Build FAISS index over English agent answers
│   └── faiss_index/                 # Saved FAISS index (index.faiss + index.pkl)
│
├── mlruns/ / mlartifacts/           # Top-level MLflow artifacts from some runs
└── src/end_to_end_nlp_project/      # Package stub (`Hello from…`)
```

---

## 4. Dataset

**File:** `Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv`

| Property | Value |
|----------|-------|
| Rows | 28,587 |
| Columns | 16 |
| Duplicates | 0 fully duplicated rows |

### Columns

| Column | Role | Notes |
|--------|------|-------|
| `subject` | Ticket subject | ~13.4% null |
| `body` | Ticket message | Always present; main NLP input |
| `answer` | Support reply | Not used as a priority feature (unavailable at inference); used for RAG |
| `type` | Incident / Change / Problem / Request | Weak priority signal |
| `queue` | 10 routing queues | **Strongest categorical driver** (Cramér’s V ≈ 0.286) |
| `priority` | Target | medium 40.3% · high 39.1% · low 20.6% |
| `language` | `en` / `de` | No priority signal; used for filtering |
| `version` | 51 / 52 / 400 | Near-constant; droppable |
| `tag_1` … `tag_8` | Multi-label topics | Sparse toward `tag_8`; tag_1–tag_5 used in model |

### Priority distribution

| Level | Count | % |
|-------|-------|---|
| medium | 11,515 | 40.28% |
| high | 11,178 | 39.10% |
| low | 5,894 | 20.62% |

### Modelling subset

English rows only → **16,338** tickets (from preprocessing logs).

---

## 5. Tech stack

### Core ML / data

| Technology | Use |
|------------|-----|
| **Python 3.12+** | Runtime |
| **uv** | Package / lockfile management (`pyproject.toml`, `uv.lock`) |
| **pandas** | Data loading & manipulation |
| **NumPy** | Feature matrices |
| **scikit-learn** | OneHotEncoder, LabelEncoder, train_test_split, RandomForest, LogisticRegression, VotingClassifier, GridSearchCV, metrics |
| **sentence-transformers** | `all-MiniLM-L6-v2` → 384-d body embeddings |
| **joblib** | Persist / load model bundle |
| **matplotlib / seaborn** | Confusion matrices & plots |
| **lightgbm / xgboost** | Declared deps (explored / planned; RF is production model) |

### Experiment tracking

| Technology | Use |
|------------|-----|
| **MLflow** | Experiments, params, metrics, models, confusion matrix artifacts |

### LLM / agents

| Technology | Use |
|------------|-----|
| **Groq** (`langchain-groq`) | Judge + agent LLM — model `openai/gpt-oss-20b` |
| **LangChain** | Prompts, structured output (Pydantic) |
| **LangGraph** | ReAct agent for disagreement logging (`create_react_agent`) |
| **Pydantic** | `ModelPrediction`, `JudgeVerdict`, `TicketFeatures` schemas |
| **python-dotenv** | Load `.env` secrets |
| **LangSmith tracing** | `LANGCHAIN_TRACING_V2=true`, project `end-to-end-nlp-project` |

### Storage / RAG

| Technology | Use |
|------------|-----|
| **DuckDB** | `feedback_tickets.duckdb` — store RF↔LLM disagreements |
| **FAISS** (`langchain-community`) | Vector store for historical answers |
| **Google Generative AI embeddings** (`langchain-google-genai`) | `gemini-embedding-001` for RAG index |

### App / UI

| Technology | Use |
|------------|-----|
| **Streamlit** | Interactive ticket priority classifier (`main.py`, `LLM-Integration/main-app.py`) |

### Env / secrets (expected in `.env`)

- `GROQ_API_KEY` — judge & agent
- `GEMINI_API_KEY` — RAG embeddings (`GOOGLE_API_KEY` derived from it)
- Optional LangSmith keys for tracing

---

## 6. Pipeline phases (as implemented)

### Phase 1 — Data analysis (done)

- Notebook: `Data Analysis /data-analysis.ipynb`
- Write-up: `Opencode/analysis.md`
- Findings: `queue` + tags + `body` semantics drive priority; drop/ignore `answer`, `language`, `version` as predictors; handle missing `subject` and sparse tags.

### Phase 2 — Preprocessing (done)

**Script:** `Pre Processing + Model Eval/preprocessing.py`

1. Filter `language == "en"`.
2. Clean `body` (strip HTML placeholders, literal `\n`/`\t`, collapse whitespace).
3. Embed with **`all-MiniLM-L6-v2`** → shape `(n, 384)`.
4. Categoricals: `queue`, `tag_1`, `tag_3`, `tag_4`, `tag_5` (NaN → `"unknown"`).
5. Stratified **80/20** train/test (`random_state=42`).
6. Frequency-cap rare categories on **train only** (`MIN_COUNT=60` → `"Other"`).
7. Fit `OneHotEncoder(handle_unknown="ignore")` on train; transform test.

**Final feature shapes (from last run):**

| Split | Shape |
|-------|-------|
| `X_train` | (13,070, 477) — 384 emb + 93 OHE |
| `X_test` | (3,268, 477) |

### Phase 3 — Model comparison & tuning (done)

**Script:** `Pre Processing + Model Eval/evaluation.py`

MLflow experiments include:

| Experiment | What was run |
|------------|--------------|
| `testingxgboost , svc and lightbgm` | Baseline RF compare + GridSearch on `n_estimators` / `max_depth` / `min_samples_split` |
| `rf_lr_ensemble` | Hard-voting RF + LogisticRegression; further RF GridSearch on leaf/features/class_weight/criterion |
| `finalmodel accuracy experimnet` | Final RF with best params + confusion matrix artifact |

### Phase 4 — Final model export (done)

**Script:** `Final Model Predictions/final.py`
**Artifact:** `final_rf_model.pkl`

**Best RandomForest hyperparameters:**

```text
n_estimators=200
max_depth=None
min_samples_split=2
min_samples_leaf=1
max_features="sqrt"
class_weight="balanced"
criterion="gini"
random_state=42
```

**Bundle contents:** `classifier`, `label_encoder`, `onehot_encoder`, `sentence_encoder`, `cat_cols`, `min_count`.

### Phase 5 — Deployment UI (done)

**Primary app:** `main.py` (Streamlit)

- User enters ticket body + selects categorical features.
- RF predicts priority from embedding + OHE.
- Groq **LLM judge** confirms or overrides (`priority: none` = agree).
- Judge also returns a **≤20-word suggested action**.
- On disagreement → **LangGraph agent** writes the ticket + LLM label into DuckDB for future retraining.

**Earlier UI:** `LLM-Integration/main-app.py` (judge pipeline without the learning agent).

### Phase 6 — LLM integration & feedback loop (done)

| File | Role |
|------|------|
| `LLM-Integration/llm-integ.py` | Batch evaluate RF + judge vs ground-truth CSV |
| `LLM-Integration/agentic.py` | ReAct agent + DuckDB `ticket_disagreements` table |
| `test.py` | Batch over `sample_test_dataset.csv` with optional agent storage |

### Phase 7 — RAG (in progress / index built)

**Script:** `RAG/rag-integ.py`

- Loads English rows with non-null `answer`.
- Indexes up to **10,000** answers as LangChain `Document`s (metadata: subject, body, type, queue, priority, tags…).
- Embeds with **Gemini `gemini-embedding-001`**.
- Saves FAISS locally under `RAG/faiss_index/`.
- **Status:** Index build script + saved index exist; full retrieval → LLM answer-generation UX is not yet wired into `main.py`.

---

## 7. Inference architecture

```text
Ticket body + (queue, tag_1, tag_3, tag_4, tag_5)
        │
        ▼
  clean_text(body)
        │
        ▼
  MiniLM encode (384-d) ──┐
                          ├── hstack → X
  OneHotEncoder cats ─────┘
        │
        ▼
  RandomForestClassifier → model_pred (high|medium|low)
        │
        ▼
  Groq LLM judge (structured JudgeVerdict)
        │
        ├─ priority == "none"  → keep model_pred
        └─ else                → override with LLM priority
                                      │
                                      ▼
                            LangGraph agent → DuckDB
                            (self-learning feedback)
        │
        ▼
  Final priority + suggested action → Streamlit UI
```

---

## 8. Current model performance

From `Pre Processing + Model Eval/run_out.log` (Final RandomForest on held-out English test set):

| Metric | Value |
|--------|-------|
| Train accuracy | 1.0000 *(overfit signal; inspect carefully)* |
| Test accuracy | **0.7867** |
| Precision (weighted) | 0.7947 |
| Precision (macro) | 0.8110 |
| Recall (weighted) | 0.7867 |
| Recall (macro) | 0.7620 |
| F1 (weighted) | 0.7851 |
| F1 (macro) | 0.7781 |

**Per-class (test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| high | 0.79 | 0.84 | 0.81 | 1,269 |
| low | 0.89 | 0.63 | 0.74 | 675 |
| medium | 0.75 | 0.82 | 0.78 | 1,324 |

Note: **low** priority has lower recall — minority class remains harder despite `class_weight="balanced"`.

---

## 9. Implementation status checklist

| Area | Status | Notes |
|------|--------|-------|
| Dataset ingested | ✅ Done | Main CSV under `Data/` |
| EDA notebook + analysis.md | ✅ Done | Descriptive only |
| Text cleaning + MiniLM embeddings | ✅ Done | English subset |
| Categorical encoding + rare capping | ✅ Done | Train-only capping |
| Model comparison / GridSearch / ensemble | ✅ Done | Logged in MLflow |
| Final RF trained & exported | ✅ Done | `final_rf_model.pkl` |
| Confusion matrix artifact | ✅ Done | PNG + MLflow |
| Streamlit inference UI | ✅ Done | `main.py` is canonical |
| LLM-as-judge | ✅ Done | Groq structured output |
| Suggested action generation | ✅ Done | In judge schema |
| Disagreement → DuckDB agent | ✅ Done | `agentic.py` |
| Batch evaluation scripts | ✅ Done | `Data/test.py`, `llm-integ.py`, `test.py` |
| FAISS RAG index build | ✅ Done | English answers, sample 10k |
| RAG retrieval in product UI | ⏳ Not integrated | Index exists; not called from `main.py` |
| German tickets in RF training | ❌ Out of scope currently | Filtered to `en` |
| FastAPI serving | ❌ Not implemented | Streamlit only |
| Automated retrain from DuckDB | ❌ Not automated | Store only |
| README documentation | ⚠️ Empty | This `project.md` is the real summary |
| Package entrypoint | ⚠️ Stub | `src/.../__init__.py` prints hello only |

---

## 10. How to run (typical)

```bash
# From repo root, with uv / .venv activated and .env populated

# Rebuild features + run final MLflow RF eval
python "Pre Processing + Model Eval/evaluation.py"

# Export / retrain final.pkl
python "Final Model Predictions/final.py"

# Streamlit app (primary product)
streamlit run main.py

# Batch RF-only eval
python Data/test.py

# Batch RF + LLM judge
python LLM-Integration/llm-integ.py

# Sample CLI with agentic feedback
python test.py

# Build / rebuild FAISS RAG index (needs GEMINI_API_KEY)
python RAG/rag-integ.py
```

**MLflow UI (when tracking server is up):** typically `http://127.0.0.1:5001` (see `run_out.log`).

---

## 11. Key design decisions

1. **English-only modelling** — simplifies tokenization; German data reserved for future multilingual models.
2. **Exclude `answer` from priority features** — would leak / unavailable at prediction time; reused for RAG.
3. **Embeddings + OHE hybrid** — semantic signal from body; strong tabular signal from `queue`/tags.
4. **LLM-as-judge override** — classical ML for speed/cost; LLM for severity reasoning and user-facing actions.
5. **Self-learning loop** — disagreements land in DuckDB for later fine-tuning / retrain datasets.
6. **Frequency capping before OHE** — reduces sparse one-hots; `handle_unknown="ignore"` for new categories at inference.
7. **Structured Pydantic outputs** — keeps judge decisions machine-parseable (`none` / `high` / `medium` / `low`).

---

## 12. Known gaps & next steps

1. Wire **RAG retrieval** into Streamlit (retrieve similar past answers; optionally draft a reply).
2. Close the loop: **retrain RF** from DuckDB disagreements + original data.
3. Address **train accuracy = 1.0** (regularization, depth limits, or out-of-fold eval).
4. Improve **low-priority recall** (resampling, threshold tuning, or cost-sensitive metrics).
5. Fix hardcoded absolute path in `preprocessing.py` → relative/`pathlib` for portability.
6. Expand README; consider FastAPI for non-UI serving.
7. Optionally train multilingual (en+de) with a multilingual encoder.

---

## 13. Related docs in this repo

| Doc | Purpose |
|-----|---------|
| `Opencode/analysis.md` | Full column-level EDA & predictive ranking |
| `Opencode/thinking.md` | How the coding agent should phase complex tasks |
| `Opencode/project.md` | **This file** — single source of truth for project status |

---

*Last summarized from the live codebase layout and scripts (data analysis, preprocessing, evaluation, final model, Streamlit app, LLM judge, agentic feedback, and RAG index builder).*
