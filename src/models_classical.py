"""
models_classical.py — TF-IDF + classical machine learning (RQ1, RQ2).

Chapter 6.2 of the thesis.

Three models, as listed in Milestone 1:
  1. Multinomial Naive Bayes — fast, simple, strong baseline on text
  2. Logistic Regression     — well-calibrated probabilities, interpretable weights
  3. Linear SVM              — often the best linear model for text classification

All three are wrapped in a scikit-learn Pipeline (TF-IDF + classifier),
because this way:
  - no information leaks from the test set into the TF-IDF fit
    (this is the most common mistake, and the one reviewers look for first!)
  - GridSearchCV can tune the vectorizer and the model together
  - the whole thing can be saved into a single .joblib file
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


# ==========================================================================
# 1) TF-IDF vectorizer
# ==========================================================================
def make_vectorizer(**overrides) -> TfidfVectorizer:
    """
    TF-IDF settings from config.py, with overridable parameters.

    What do they mean?
      ngram_range=(1,2) : single-word AND two-word expressions. The bigram
                          "not clean" is worth gold in a sentiment task.
      min_df=3          : a word occurring in fewer than 3 documents is noise
                          (typos, unique names) -> dropped
      max_df=0.9        : a word present in almost every review ("hotel")
                          carries no discriminating information
      sublinear_tf=True : 1 + log(tf) instead of the raw frequency. Dampens
                          the effect of words repeated many times.
    """
    params = dict(
        max_features=CFG.tfidf_max_features,
        ngram_range=CFG.tfidf_ngram_range,
        min_df=CFG.tfidf_min_df,
        max_df=CFG.tfidf_max_df,
        sublinear_tf=CFG.tfidf_sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


# ==========================================================================
# 2) Models
# ==========================================================================
def get_classifier(name: str, class_weight: str | None = "balanced"):
    """
    class_weight="balanced": errors on the rarer class (typically neutral)
    get a larger weight. On an imbalanced 3-class task this improves the
    macro-F1 considerably. MultinomialNB does not support it, there the
    class_prior stays at its default.
    """
    name = name.lower()
    if name in {"nb", "naive_bayes", "multinomialnb"}:
        return MultinomialNB(alpha=0.5)
    if name in {"logreg", "logistic_regression", "lr"}:
        return LogisticRegression(
            max_iter=2000, C=1.0, class_weight=class_weight,
            random_state=CFG.random_seed,
        )
    if name in {"svm", "linear_svm", "linearsvc"}:
        return LinearSVC(
            C=1.0, class_weight=class_weight,
            random_state=CFG.random_seed, max_iter=5000,
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(model_name: str, vectorizer_kwargs: dict | None = None,
                   class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline([
        ("tfidf", make_vectorizer(**(vectorizer_kwargs or {}))),
        ("clf", get_classifier(model_name, class_weight=class_weight)),
    ])


MODEL_NAMES = ["nb", "logreg", "svm"]


# ==========================================================================
# 3) Training and evaluation
# ==========================================================================
def train_and_evaluate_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "sentiment",
    model_names: list[str] | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Train all three models on the training set and evaluate them on the
    VALIDATION set. The test set is NOT touched YET at this point — it is
    used only once, after the final model selection.

    Returns: (summary table, {model name: fitted pipeline})
    """
    from .evaluation import compute_metrics

    model_names = model_names or MODEL_NAMES
    rows, fitted = [], {}

    X_tr, y_tr = train_df[text_col], train_df[label_col].astype(str)
    X_va, y_va = val_df[text_col], val_df[label_col].astype(str)

    for name in model_names:
        log.info("Training: %s (%d samples)", name, len(X_tr))
        pipe = build_pipeline(name)
        pipe.fit(X_tr, y_tr)

        m = compute_metrics(y_va, pipe.predict(X_va))
        m["model"] = name
        m["n_train"] = len(X_tr)
        m["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
        rows.append(m)
        fitted[name] = pipe

        if save:
            save_model(pipe, f"tfidf_{name}")

    summary = (pd.DataFrame(rows)
               .set_index("model")
               .sort_values("f1_macro", ascending=False))
    return summary, fitted


def save_model(pipe: Pipeline, name: str) -> Path:
    import joblib
    PATHS.ensure()
    path = PATHS.models / f"{name}.joblib"
    joblib.dump(pipe, path)
    log.info("Model saved: %s", path)
    return path


def load_model(name: str) -> Pipeline:
    import joblib
    return joblib.load(PATHS.models / f"{name}.joblib")


# ==========================================================================
# 4) ABLATION STUDY — answers RQ2 directly
# ==========================================================================
def preprocessing_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "logreg",
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """
    RQ2: "How do preprocessing and feature representation affect the
    performance of the classical models?"

    This function gives the QUANTITATIVE answer: it runs the same model with
    different preprocessing and TF-IDF settings and returns a table. The
    table can go into the thesis as it is.

    Prerequisite: train_df / val_df contain the 'text' (raw) and
    'text_clean' (full cleaning) columns.
    """
    from .evaluation import compute_metrics
    from .preprocessing import clean_text_classic

    variants: list[tuple[str, str, dict]] = [
        # (name, which text column, TF-IDF overrides)
        ("raw text, unigram",                "text",       {"ngram_range": (1, 1)}),
        ("raw text, uni+bigram",             "text",       {"ngram_range": (1, 2)}),
        ("full cleaning, unigram",           "text_clean", {"ngram_range": (1, 1)}),
        ("full cleaning, uni+bigram",        "text_clean", {"ngram_range": (1, 2)}),
        ("full cleaning, uni+bi+trigram",    "text_clean", {"ngram_range": (1, 3)}),
        ("full cleaning, sublinear_tf off",  "text_clean", {"sublinear_tf": False}),
        ("full cleaning, max_feat=10k",      "text_clean", {"max_features": 10_000}),
    ]

    # Extra variant: stopword filtering WITHOUT keeping the negation words.
    # This often makes things measurably worse -> a great observation for the thesis.
    for split_df in (train_df, val_df):
        if "text_nonegation" not in split_df.columns:
            split_df["text_nonegation"] = split_df["text"].map(
                lambda t: clean_text_classic(t, keep_negations=False)
            )
    variants.append(("cleaning, negation words dropped", "text_nonegation", {}))

    rows = []
    for label, col, kw in variants:
        pipe = build_pipeline(model_name, vectorizer_kwargs=kw)
        pipe.fit(train_df[col], train_df[label_col].astype(str))
        m = compute_metrics(val_df[label_col].astype(str), pipe.predict(val_df[col]))
        m["variant"] = label
        m["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
        rows.append(m)
        log.info("%-38s macro-F1 = %.4f", label, m["f1_macro"])

    return (pd.DataFrame(rows)
            .set_index("variant")
            .sort_values("f1_macro", ascending=False))


# ==========================================================================
# 5) INTERPRETABILITY — which words decide?
# ==========================================================================
def top_features_per_class(pipe: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Extract the highest-weighted features of the linear model per class.

    This helps directly with RQ3 ("which aspects are tied to negative
    reviews?") and gives a great, easily interpretable table for the thesis:
    you will see that e.g. the words "dirty", "noisy", "smell" drive the
    negative class.

    Works only with LogisticRegression / LinearSVC (for NB the
    log-probabilities are different and need separate handling).
    """
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("This model is not linear (it has no coef_ attribute).")

    feature_names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "sentiment": cls,
                "rank": rank,
                "feature": feature_names[idx],
                "weight": float(coefs[idx]),
            })
    return pd.DataFrame(rows)
