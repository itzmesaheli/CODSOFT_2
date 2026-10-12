"""
CODSOFT Task 2 - Credit Card Fraud Detection
--------------------------------------------
Detects fraudulent credit card transactions with Logistic Regression,
Decision Tree and Random Forest, handling the extreme class imbalance.

Dataset: put the CSV file(s) in ./data/ . Works with either format:
    - creditcard.csv                     (Kaggle ULB dataset, target "Class")
    - fraudTrain.csv + fraudTest.csv     (Kaggle simulated dataset, target "is_fraud")

Usage:
    pip install pandas scikit-learn matplotlib joblib
    python fraud_detector.py
    python fraud_detector.py --score data/new_transactions.csv
"""

import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data")
OUT_DIR = Path("output")
MODEL_PATH = OUT_DIR / "fraud_model.joblib"
RANDOM_STATE = 42
MAX_LEGIT_TRAIN = 60_000          # legitimate rows kept for training (all fraud rows are kept)

TARGET_CANDIDATES = ["class", "is_fraud", "isfraud", "fraud"]
DROP_COLUMNS = ["unnamed: 0", "index", "id", "cc_num", "trans_num", "unix_time",
                "trans_date_trans_time", "dob", "zip", "first", "last", "street"]


# ---------------------------------------------------------------- data loading
def load_data():
    """Return (train_df, test_df_or_None)."""
    train_files = sorted(DATA_DIR.glob("fraudTrain*.csv"))
    test_files = sorted(DATA_DIR.glob("fraudTest*.csv"))
    if train_files and test_files:
        return pd.read_csv(train_files[0]), pd.read_csv(test_files[0])
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit("No CSV found. Put the fraud dataset inside the 'data' folder.")
    return pd.read_csv(csvs[0]), None


def find_target(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() in TARGET_CANDIDATES:
            return col
    raise SystemExit(f"Could not find the fraud column. Columns: {list(df.columns)}")


def drop_bad_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove rows whose fraud label is missing or not a number (blank or cut-off lines)."""
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    bad = int(df[target].isna().sum())
    if bad:
        print(f"    Note: dropped {bad:,} rows with a missing or invalid '{target}' value")
    return df.dropna(subset=[target]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def prepare_features(df: pd.DataFrame, target) -> pd.DataFrame:
    """Engineer time/distance features, drop ID-like and high-cardinality columns."""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    if "trans_date_trans_time" in cols:
        t = pd.to_datetime(df[cols["trans_date_trans_time"]], errors="coerce")
        df["hour"], df["dayofweek"], df["month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
        if "dob" in cols:
            df["age"] = (t - pd.to_datetime(df[cols["dob"]], errors="coerce")).dt.days / 365.25
    if all(k in cols for k in ("lat", "long", "merch_lat", "merch_long")):
        df["distance_km"] = haversine_km(
            df[cols["lat"]], df[cols["long"]], df[cols["merch_lat"]], df[cols["merch_long"]]
        )

    drop = [c for c in df.columns if c.lower() in DROP_COLUMNS]
    if target:
        drop.append(target)
    X = df.drop(columns=drop, errors="ignore")
    for col in X.select_dtypes(exclude="number").columns:
        if X[col].nunique() > 50:                    # merchant, city, job, ...
            X = X.drop(columns=col)
    return X


def build_pipeline(X: pd.DataFrame, clf) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def downsample_legit(X: pd.DataFrame, y: pd.Series):
    """Keep every fraud row and at most MAX_LEGIT_TRAIN legitimate rows."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    if len(legit_idx) > MAX_LEGIT_TRAIN:
        legit_idx = rng.choice(legit_idx, MAX_LEGIT_TRAIN, replace=False)
    keep = np.concatenate([fraud_idx, legit_idx])
    rng.shuffle(keep)
    return X.iloc[keep], y.iloc[keep]


# ------------------------------------------------------------------- training
def train_and_evaluate() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: load and split
    train_df, test_df = load_data()
    target = find_target(train_df)
    train_df = drop_bad_rows(train_df, target)
    if test_df is not None:
        test_df = drop_bad_rows(test_df, target)
    if test_df is None:
        y_all = train_df[target].astype(int)
        X_all = prepare_features(train_df, target)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
        )
    else:
        X_tr, y_tr = prepare_features(train_df, target), train_df[target].astype(int)
        X_te, y_te = prepare_features(test_df, target), test_df[target].astype(int)
        X_te = X_te[X_tr.columns]

    print(f"[1] Train: {len(X_tr):,} transactions ({int(y_tr.sum()):,} fraud, {y_tr.mean():.3%})")
    print(f"    Test : {len(X_te):,} transactions ({int(y_te.sum()):,} fraud, {y_te.mean():.3%})")
    print(f"    Features used: {list(X_tr.columns)}\n")

    # Step 2: handle imbalance (downsample the legitimate class for training only)
    X_fit, y_fit = downsample_legit(X_tr, y_tr)
    print(f"[2] Training on {len(X_fit):,} rows after downsampling "
          f"({int(y_fit.sum()):,} fraud, {int((y_fit == 0).sum()):,} legitimate)")
    print("    The test set is left untouched, so scores reflect real-world imbalance.\n")

    # Step 3: compare models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }
    results = {}
    print("[3] Model comparison on the full test set (fraud = positive class)")
    print(f"    {'Model':<22}{'Prec':>8}{'Recall':>8}{'F1':>8}{'PR-AUC':>8}{'ROC-AUC':>9}")
    for name, clf in candidates.items():
        start = time.time()
        pipe = build_pipeline(X_fit, clf).fit(X_fit, y_fit)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        scores = (
            precision_score(y_te, pred, zero_division=0), recall_score(y_te, pred),
            f1_score(y_te, pred), average_precision_score(y_te, proba), roc_auc_score(y_te, proba),
        )
        results[name] = (scores[3], pipe, pred)
        print(f"    {name:<22}" + "".join(f"{s:>8.4f}" for s in scores[:4])
              + f"{scores[4]:>9.4f}  ({time.time() - start:.0f}s)")

    best_name = max(results, key=lambda k: results[k][0])
    _, best_pipe, best_pred = results[best_name]
    tn, fp, fn, tp = confusion_matrix(y_te, best_pred).ravel()
    print(f"\n[4] Best model (by PR-AUC): {best_name}")
    print(f"    Caught {tp:,} of {tp + fn:,} frauds | {fp:,} false alarms among {tn + fp:,} legitimate transactions\n")
    print(classification_report(y_te, best_pred, target_names=["legitimate", "fraud"], digits=4))

    # Step 5: plots
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, best_pred, display_labels=["legit", "fraud"], ax=ax, cmap="Blues",
        colorbar=False, values_format="d",
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, pipe, _) in results.items():
        PrecisionRecallDisplay.from_estimator(pipe, X_te, y_te, ax=ax, name=name)
    ax.set_title("Precision-Recall curves")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "precision_recall_curves.png", dpi=150)
    plt.close(fig)

    clf = best_pipe.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in best_pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(clf.feature_importances_ if hasattr(clf, "feature_importances_")
                    else np.abs(clf.coef_[0]), index=names)
    top = imp.sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color="#38bdf8")
    ax.set_title(f"Top features - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("[5] Top fraud signals:", ", ".join(top.index[::-1][:5]))

    # Step 6: save
    joblib.dump({"model": best_pipe, "columns": list(X_tr.columns)}, MODEL_PATH)
    print(f"[6] Model saved to {MODEL_PATH}")


# ----------------------------------------------------------------- prediction
def score_file(csv_path: str) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run the script without --score first.")
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)
    target = next((c for c in df.columns if c.strip().lower() in TARGET_CANDIDATES), None)
    X = prepare_features(df, target)[bundle["columns"]]
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba.round(4)
    out["predicted_fraud"] = (proba >= 0.5).astype(int)
    dest = OUT_DIR / "scored_transactions.csv"
    out.to_csv(dest, index=False)
    print(f"Scored {len(out):,} transactions -> {dest}")
    print(f"Flagged as fraud: {int(out['predicted_fraud'].sum()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODSOFT Task 2 - Credit Card Fraud Detection")
    parser.add_argument("--score", metavar="CSV", help="Score transactions from a CSV file")
    args = parser.parse_args()

    if args.score:
        score_file(args.score)
    else:
        train_and_evaluate()
