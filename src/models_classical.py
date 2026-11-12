"""TF-IDF and classical classifiers trained on the BERT labels.

The student models are fast, run on a CPU and their weights show which terms drive a
class. Their test score measures agreement with the BERT labels, not accuracy; accuracy
is measured on the gold sample. TF-IDF and the classifier form one pipeline, so the
vocabulary is built from the training data only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import CFG, PATHS

log = logging.getLogger(__name__)

MODEL_NAMES = ["nb", "logreg", "svm"]
DEFAULT_LABEL_COL = "label_final"


def make_vectorizer(**overrides) -> TfidfVectorizer:
    """
    TF-IDF with the config.py settings:
      ngram_range=(1,2) - words and word pairs ("not clean" exists only as a bigram)
      min_df=3          - a word rarer than 3 documents is noise
      max_df=0.9        - a word present almost everywhere ("hotel") is uninformative
      sublinear_tf=True - 1 + log(tf): damps words repeated many times
    """
    params = dict(max_features=CFG.tfidf_max_features, ngram_range=CFG.tfidf_ngram_range,
                  min_df=CFG.tfidf_min_df, max_df=CFG.tfidf_max_df,
                  sublinear_tf=CFG.tfidf_sublinear_tf, strip_accents="unicode", lowercase=True)
    params.update(overrides)
    return TfidfVectorizer(**params)


def get_classifier(name: str, class_weight: str | None = "balanced"):
    """
    class_weight="balanced": errors on the rarer class get a larger weight
    (MultinomialNB does not support it).
    """
    name = name.lower()
    if name in {"nb", "naive_bayes", "multinomialnb"}:
        return MultinomialNB(alpha=0.5)
    if name in {"logreg", "logistic_regression", "lr"}:
        return LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight,
                                  random_state=CFG.random_seed)
    if name in {"svm", "linear_svm", "linearsvc"}:
        return LinearSVC(C=1.0, class_weight=class_weight, random_state=CFG.random_seed, max_iter=5000)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(model_name: str, vectorizer_kwargs: dict | None = None,
                   class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer(**(vectorizer_kwargs or {}))),
                     ("clf", get_classifier(model_name, class_weight=class_weight))])


def train_and_evaluate_all(train_df: pd.DataFrame, val_df: pd.DataFrame, text_col: str = "text_clean",
                           label_col: str = DEFAULT_LABEL_COL, model_names: list[str] | None = None,
                           save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Train all three models (train) and evaluate them (val). The test set is
    touched only once, after the final model selection.

    Returns: (summary table, {model name: fitted pipeline})
    """
    from .evaluation import compute_metrics

    rows, fitted = [], {}
    x_train, y_train = train_df[text_col], train_df[label_col].astype(str)
    x_val, y_val = val_df[text_col], val_df[label_col].astype(str)
    for name in model_names or MODEL_NAMES:
        log.info("Training: %s (%d samples)", name, len(x_train))
        pipe = build_pipeline(name).fit(x_train, y_train)
        m = compute_metrics(y_val, pipe.predict(x_val))
        m.update(model=name, n_train=len(x_train), n_features=len(pipe.named_steps["tfidf"].vocabulary_))
        rows.append(m)
        fitted[name] = pipe
        if save:
            save_model(pipe, f"tfidf_{name}_{label_col}")
    summary = pd.DataFrame(rows).set_index("model").sort_values("f1_macro", ascending=False)
    return summary, fitted


def save_model(pipe: Pipeline, name: str) -> Path:
    import joblib
    PATHS.ensure()
    path = PATHS.models / f"{name}.joblib"
    joblib.dump(pipe, path)
    log.info("Model saved: %s", path.name)
    return path


def load_model(name: str) -> Pipeline:
    import joblib
    return joblib.load(PATHS.models / f"{name}.joblib")


def preprocessing_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame, model_name: str = "logreg",
                           label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """
    Ablation study: the same model with different preprocessing and TF-IDF
    settings. Shows how much each design decision is worth.
    """
    from .evaluation import compute_metrics
    from .preprocessing import clean_text_classic

    variants: list[tuple[str, str, dict]] = [
        ("raw text, unigram", "text", {"ngram_range": (1, 1)}),
        ("raw text, uni+bigram", "text", {"ngram_range": (1, 2)}),
        ("full cleaning, unigram", "text_clean", {"ngram_range": (1, 1)}),
        ("full cleaning, uni+bigram", "text_clean", {"ngram_range": (1, 2)}),
        ("full cleaning, uni+bi+trigram", "text_clean", {"ngram_range": (1, 3)}),
        ("full cleaning, sublinear_tf off", "text_clean", {"sublinear_tf": False}),
        ("full cleaning, max_features=10k", "text_clean", {"max_features": 10_000}),
    ]
    train_df, val_df = train_df.copy(), val_df.copy()
    for part in (train_df, val_df):
        part["text_nonegation"] = part["text"].map(lambda t: clean_text_classic(t, keep_negations=False))
    variants.append(("cleaning, negation words dropped", "text_nonegation", {}))

    rows = []
    for name, col, kwargs in variants:
        pipe = build_pipeline(model_name, vectorizer_kwargs=kwargs)
        pipe.fit(train_df[col], train_df[label_col].astype(str))
        m = compute_metrics(val_df[label_col].astype(str), pipe.predict(val_df[col]))
        m.update(variant=name, n_features=len(pipe.named_steps["tfidf"].vocabulary_))
        rows.append(m)
        log.info("%-36s macro-F1 = %.4f", name, m["f1_macro"])
    return pd.DataFrame(rows).set_index("variant").sort_values("f1_macro", ascending=False)


def top_features_per_class(pipe: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Highest-weighted terms of the linear model per class (LogReg / LinearSVC).
    """
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("This model is not linear (no coef_ attribute).")
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.ndim > 1 and clf.coef_.shape[0] > 1 else clf.coef_[0]
        for rank, idx in enumerate(np.argsort(coefs)[-top_n:][::-1], start=1):
            rows.append({"sentiment": cls, "rank": rank, "feature": names[idx], "weight": float(coefs[idx])})
    return pd.DataFrame(rows)
