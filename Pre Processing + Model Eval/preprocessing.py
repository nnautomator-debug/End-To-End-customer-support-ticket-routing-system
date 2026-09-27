import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer



CD = "/Users/ashriths/Library/Mobile Documents/com~apple~CloudDocs/Repositries/End To End NLP Project"
df = pd.read_csv(f"{CD}/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

df_1 = df[df["language"] == "en"].reset_index(drop=True)
print(f"[1] df_1 (English only): {df_1.shape[0]} rows, {df_1.shape[1]} cols")



def clean_text(text: str) -> str:
    """Strip HTML tags, escaped-newline sequences, and excess whitespace.
    Apostrophes, exclamation marks and normal punctuation are preserved
    so the sentence-transformer tokenizer can use them."""
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df_1["body_clean"] = df_1["body"].apply(clean_text)
sample_len = len(df_1["body_clean"].iloc[0])
print(f"[2] body cleaned — sample cleaned length: {sample_len} chars")

ST_MODEL = "all-MiniLM-L6-v2"

print(f"[3] Loading sentence-transformers model: {ST_MODEL} ...")
encoder = SentenceTransformer(ST_MODEL)
embeddings = encoder.encode(
    df_1["body_clean"].tolist(),
    show_progress_bar=True,
    batch_size=256,
)
print(f"[3] Embeddings shape: {embeddings.shape}")

CAT_COLS = ["queue", "tag_1", "tag_3", "tag_4", "tag_5"]

cat_raw = df_1[CAT_COLS].fillna("unknown").astype(str)
print(f"[4] Raw categorical shape (no encoding yet): {cat_raw.shape}")

X = np.hstack([embeddings, cat_raw.to_numpy()])
y = df_1["priority"]

print(f"[5] X shape (embeddings + unencoded cats): {X.shape}")
print(f"[5] y shape:                      {y.shape}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

emb_dim = embeddings.shape[1]
cat_train = pd.DataFrame(X_train_raw[:, emb_dim:], columns=CAT_COLS)
cat_test = pd.DataFrame(X_test_raw[:, emb_dim:], columns=CAT_COLS)
emb_train = X_train_raw[:, :emb_dim].astype(np.float32)
emb_test = X_test_raw[:, :emb_dim].astype(np.float32)

MIN_COUNT = 60
total_rare = 0
for col in CAT_COLS:
    counts = cat_train[col].value_counts()
    rare = counts[counts < MIN_COUNT].index
    cat_train[col] = cat_train[col].replace(rare, "Other")
    total_rare += len(rare)
print(f"[6] Frequency capping applied on X_train only: {total_rare} rare categories (<{MIN_COUNT}) relabelled to 'Other'")

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cat_train_enc = ohe.fit_transform(cat_train)
cat_test_enc = ohe.transform(cat_test)
print(f"[6] One-hot encoded on capped X_train cats: {cat_train_enc.shape} cols "
      f"(test left untouched, shape {cat_test_enc.shape})")

X_train = np.hstack([emb_train, cat_train_enc])
X_test = np.hstack([emb_test, cat_test_enc])

print()
print("=" * 50)
print("Final shapes")
print("=" * 50)
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")
print("=" * 50)
print("y_train distribution:\n", y_train.value_counts(normalize=True).round(4).to_string())
print("y_test  distribution:\n", y_test.value_counts(normalize=True).round(4).to_string())

