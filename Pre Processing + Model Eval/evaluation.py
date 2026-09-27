import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing
from preprocessing import X_train, X_test, y_train, y_test

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "testingxgboost , svc and lightbgm"
ENSEMBLE_EXPERIMENT = "rf_lr_ensemble"
FINAL_EXPERIMENT = "finalmodel accuracy experimnet"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
    silent=True,
)


def compare_models():
    mlflow.set_experiment(EXPERIMENT_NAME)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)

    tree_models = {
        "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=42),
    }
    non_tree_models = {}

    pipelines = {
        "non_tree": (non_tree_models, y_train, y_test),
        "tree": (tree_models, y_train_le, y_test_le),
    }

    for pipeline_name, (models, ytr, yte) in pipelines.items():
        for model_name, clf in models.items():
            with mlflow.start_run(run_name=f"{model_name} [{pipeline_name}]") as run:
                mlflow.log_params({"model_name": model_name, "pipeline_type": pipeline_name})

                clf.fit(X_train, ytr)
                y_pred = clf.predict(X_test)

                mlflow.log_metrics(
                    {
                        "train_accuracy": clf.score(X_train, ytr),
                        "test_accuracy": accuracy_score(yte, y_pred),
                        "precision_weighted": precision_score(yte, y_pred, average="weighted", zero_division=0),
                        "recall_weighted": recall_score(yte, y_pred, average="weighted", zero_division=0),
                        "f1_weighted": f1_score(yte, y_pred, average="weighted", zero_division=0),
                        "f1_macro": f1_score(yte, y_pred, average="macro", zero_division=0),
                    }
                )

                report = classification_report(yte, y_pred, target_names=le.classes_.tolist())
                with tempfile.NamedTemporaryFile("w", suffix="_classification_report.txt", delete=False) as f:
                    f.write(report)
                    report_path = f.name
                mlflow.log_artifact(report_path, artifact_path="metrics")
                os.unlink(report_path)

                print(f"[MLflow] run '{run.info.run_name}' logged "
                      f"(run_id={run.info.run_id})")


## Hyper Parameter Tuning
def hyper_params():
    mlflow.set_experiment(EXPERIMENT_NAME)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }

    with mlflow.start_run(run_name="RF Hyperparameter Tuning"):
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train_le)

        best_score = grid_search.score(X_test, y_test_le)
        y_pred = grid_search.predict(X_test)

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        print(f"Test score: {best_score:.3f}")
        print(f"Test accuracy:       {accuracy_score(y_test_le, y_pred):.4f}")
        print(f"Precision (weighted): {precision_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Precision (macro):    {precision_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall (weighted):    {recall_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall (macro):       {recall_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1 (weighted):        {f1_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1 (macro):           {f1_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print()
        print(classification_report(y_test_le, y_pred, target_names=le.classes_.tolist()))

        mlflow.log_params({
            "best_params": str(grid_search.best_params_),
            "cv_folds": 5,
            "scoring": "accuracy",
        })
        mlflow.log_metrics({
            "best_cv_score": grid_search.best_score_,
            "test_accuracy": best_score,
            "precision_weighted": precision_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test_le, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_test_le, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test_le, y_pred, average="macro", zero_division=0),
        })

        return grid_search.best_estimator_


## Ensemble: RandomForest + LogisticRegression via sklearn VotingClassifier
def ensemble_experiment():
    mlflow.set_experiment(ENSEMBLE_EXPERIMENT)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        n_jobs=-1, random_state=42,
    )
    lr = LogisticRegression(C=1.0, solver="saga", penalty="l1", random_state=42)
    vc = VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="hard")

    with mlflow.start_run(run_name="RF+LR Voting Ensemble (hard)"):
        mlflow.log_params({
            "rf_n_estimators": rf.n_estimators,
            "rf_max_depth": rf.max_depth,
            "rf_min_samples_split": rf.min_samples_split,
            "lr_C": lr.C,
            "lr_solver": lr.solver,
            "lr_penalty": lr.penalty,
            "ensemble_voting": "hard",
        })

        vc.fit(X_train, y_train_le)
        y_pred = vc.predict(X_test)

        mlflow.log_metrics({
            "train_accuracy": vc.score(X_train, y_train_le),
            "test_accuracy": accuracy_score(y_test_le, y_pred),
            "precision_weighted": precision_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test_le, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_test_le, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test_le, y_pred, average="macro", zero_division=0),
        })

        print(f"Ensemble (RF+LR hard voting) test accuracy: {accuracy_score(y_test_le, y_pred):.4f}")
        print(f"Train accuracy: {vc.score(X_train, y_train_le):.4f}")
        print()
        print(f"Precision (weighted): {precision_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Precision (macro):    {precision_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall (weighted):    {recall_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall (macro):       {recall_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1 (weighted):        {f1_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1 (macro):           {f1_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print()
        print(classification_report(y_test_le, y_pred, target_names=le.classes_.tolist()))

        report = classification_report(y_test_le, y_pred, target_names=le.classes_.tolist())
        with tempfile.NamedTemporaryFile("w", suffix="_classification_report.txt", delete=False) as f:
            f.write(report)
            report_path = f.name
        mlflow.log_artifact(report_path, artifact_path="metrics")
        os.unlink(report_path)

        print(f"[MLflow] run 'RF+LR Voting Ensemble (hard)' logged "
              f"(run_id={mlflow.active_run().info.run_id})")

    return vc


## RF GridSearchCV: leaf size / features / class weight / criterion
def ensembles_test():
    mlflow.set_experiment(ENSEMBLE_EXPERIMENT)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        n_jobs=-1, random_state=42,
    )

    param_grid = {
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
        "criterion": ["gini", "entropy"],
    }

    with mlflow.start_run(run_name="RF GridSearch (leaf/features/class_weight/criterion)"):
        mlflow.log_params({
            "rf_n_estimators": rf.n_estimators,
            "rf_max_depth": rf.max_depth,
            "rf_min_samples_split": rf.min_samples_split,
            "grid": str(param_grid),
            "cv_folds": 5,
            "scoring": "accuracy",
        })

        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train_le)

        best_score = grid_search.score(X_test, y_test_le)
        y_pred = grid_search.predict(X_test)

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test score: {best_score:.4f}")
        print(f"Test accuracy:       {accuracy_score(y_test_le, y_pred):.4f}")
        print(f"Precision (weighted): {precision_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Precision (macro):    {precision_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall (weighted):    {recall_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall (macro):       {recall_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1 (weighted):        {f1_score(y_test_le, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1 (macro):           {f1_score(y_test_le, y_pred, average='macro', zero_division=0):.4f}")
        print()
        print(classification_report(y_test_le, y_pred, target_names=le.classes_.tolist()))

        mlflow.log_params({
            "best_min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
            "best_max_features": grid_search.best_params_["max_features"],
            "best_class_weight": str(grid_search.best_params_["class_weight"]),
            "best_criterion": grid_search.best_params_["criterion"],
        })
        mlflow.log_metrics({
            "best_cv_score": grid_search.best_score_,
            "test_accuracy": best_score,
            "precision_weighted": precision_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test_le, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_test_le, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test_le, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test_le, y_pred, average="macro", zero_division=0),
        })

        report = classification_report(y_test_le, y_pred, target_names=le.classes_.tolist())
        with tempfile.NamedTemporaryFile("w", suffix="_classification_report.txt", delete=False) as f:
            f.write(report)
            report_path = f.name
        mlflow.log_artifact(report_path, artifact_path="metrics")
        os.unlink(report_path)

        print(f"[MLflow] run 'RF GridSearch (leaf/features/class_weight/criterion)' logged "
              f"(run_id={mlflow.active_run().info.run_id})")

    return grid_search.best_estimator_


## Final model: RandomForest with the best GridSearch params
def final_accuracy():
    mlflow.set_experiment(FINAL_EXPERIMENT)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)
    class_names = le.classes_.tolist()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        criterion="gini",
        n_jobs=-1,
        random_state=42,
    )

    with mlflow.start_run(run_name="Final RandomForest (Best Params)") as run:
        mlflow.log_params(
            {
                "n_estimators": rf.n_estimators,
                "max_depth": rf.max_depth,
                "min_samples_split": rf.min_samples_split,
                "min_samples_leaf": rf.min_samples_leaf,
                "max_features": rf.max_features,
                "class_weight": str(rf.class_weight),
                "criterion": rf.criterion,
                "cv_folds": 5,
                "scoring": "accuracy",
            }
        )

        rf.fit(X_train, y_train_le)
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
        mlflow.log_metrics(scores)

        report_dict = classification_report(
            y_test_le, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.drop(index="accuracy", errors="ignore")
        report_df["class"] = report_df.index
        report_df = report_df[["class", "precision", "recall", "f1-score", "support"]]

        conf_matrix = confusion_matrix(y_test_le, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Final RandomForest Confusion Matrix")

        cm_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "final_rf_confusion_matrix.png"
        )
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        report_txt = classification_report(y_test_le, y_pred, target_names=class_names, zero_division=0)
        with tempfile.NamedTemporaryFile("w", suffix="_classification_report.txt", delete=False) as f:
            f.write(report_txt)
            report_txt_path = f.name
        mlflow.log_artifact(report_txt_path, artifact_path="metrics")
        os.unlink(report_txt_path)

        with tempfile.NamedTemporaryFile("w", suffix="_classification_report.csv", delete=False) as f:
            report_df.to_csv(f, index=False)
            report_csv_path = f.name
        mlflow.log_artifact(report_csv_path, artifact_path="metrics")
        os.unlink(report_csv_path)

        print()
        print("=" * 62)
        print("Final RandomForest (Best GridSearch Params) — Results")
        print("=" * 62)
        for k, v in scores.items():
            print(f"  {k:20s}: {v:.4f}")
        print()
        print("Per-class precision / recall / f1 (weighted & macro included):")
        print(report_df.to_string(index=False))
        print()
        print(f"Confusion matrix saved → {cm_path}")
        print(f"[MLflow] run '{run.info.run_name}' logged (run_id={run.info.run_id})")
        print("=" * 62)

    return rf


if __name__ == "__main__":
    final_accuracy()
    print(f"MLflow UI → {TRACKING_URI}")
