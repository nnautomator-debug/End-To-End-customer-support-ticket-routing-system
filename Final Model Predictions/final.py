import joblib
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pre Processing + Model Eval"))
import preprocessing
from preprocessing import (
    X_train,
    X_test,
    y_train,
    y_test,
    ohe,
    encoder,
    CAT_COLS,
    MIN_COUNT,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "final_rf_model.pkl")

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "criterion": "gini",
    "random_state": 42,
}


def train_random_forest():
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)
    rf = RandomForestClassifier(n_jobs=-1, **BEST_PARAMS)
    rf.fit(X_train, y_train_le)
    return rf, le, y_train_le, y_test_le


def save_artifacts(rf, le, path=MODEL_PATH):
    artifacts = {
        "classifier": rf,
        "label_encoder": le,
        "onehot_encoder": ohe,
        "sentence_encoder": encoder,
        "cat_cols": CAT_COLS,
        "min_count": MIN_COUNT,
    }
    joblib.dump(artifacts, path)
    emb_dim = encoder.get_sentence_embedding_dimension()
    ohe_cols = rf.n_features_in_ - emb_dim
    print(f"[export] artifacts saved -> {path}")
    print(f"[export] feature count: {rf.n_features_in_} "
          f"({emb_dim} embedding + {ohe_cols} one-hot)")


def train_and_evaluate(export=True):
    rf, le, y_train_le, y_test_le = train_random_forest()
    y_pred = rf.predict(X_test)

    scores = {
        "train_accuracy": rf.score(X_train, y_train_le),
        "test_accuracy": accuracy_score(y_test_le, y_pred),
        "precision_weighted": precision_score(y_test_le, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test_le, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_test_le, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_test_le, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test_le, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test_le, y_pred, average="macro", zero_division=0),
    }

    print("=" * 56)
    print("Final.py - RFC on preprocessing data")
    print("=" * 56)
    for k, v in scores.items():
        print(f"  {k:20s}: {v:.4f}")
    print()
    print(classification_report(y_test_le, y_pred, target_names=le.classes_.tolist(), zero_division=0))
    print("=" * 56)

    if export:
        save_artifacts(rf, le)
    return rf, scores


if __name__ == "__main__":
    train_and_evaluate()