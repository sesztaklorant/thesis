"""
evaluation.py — Metrics, confusion matrix, error analysis.

Chapter 8 of the thesis ("Data Split and Evaluation").

Every model (classical AND BERT) is evaluated with THESE SAME functions —
this is the only way to make the RQ1 comparison fair.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import CFG, PATHS

log = logging.getLogger(__name__)


# ==========================================================================
# 1) METRICS
# ==========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """
    The metrics promised in the thesis, in one dictionary.

    Why macro AND weighted F1?
      - weighted: frequent classes count proportionally more. If the positive
        class is 70% of the data, even a "call everything positive" model
        gets a high weighted score -> MISLEADING.
      - macro: every class counts equally. This shows whether the model has
        really learned the hard neutral class.

      Your MAIN metric should be MACRO-F1, and say so in the thesis.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f_mac,
        "precision_weighted": p_wt,
        "recall_weighted": r_wt,
        "f1_weighted": f_wt,
    }


def baseline_metrics(y_true, labels: list[str] | None = None) -> pd.DataFrame:
    """
    TRIVIAL BASELINES — the numbers every one of your models has to beat.

    Why is this indispensable on imbalanced data?
    ---------------------------------------------
    If 64.5% of your data is positive, a model that says positive to
    EVERYTHING reaches 64.5% accuracy without having learned anything.
    An accuracy of 0.70 in the thesis without context sounds impressive —
    yet it is barely better than nothing.

    So this should be the FIRST ROW of the results table of the thesis.
    Every other number is to be interpreted relative to it.

    Three baselines:
      • majority class — always predicts the most frequent class
      • stratified random — draws according to the class proportions
      • uniform random — 1/3–1/3–1/3
    """
    labels = labels or list(CFG.labels)
    y = np.asarray(y_true).astype(str)
    rng = np.random.default_rng(CFG.random_seed)

    props = pd.Series(y).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    majority = props.idxmax()

    rows = {}

    # 1) Always the majority class
    rows[f"majority class ('{majority}')"] = compute_metrics(
        y, np.full(len(y), majority))

    # 2) Stratified random: draws according to the class proportions
    rows["stratified random"] = compute_metrics(
        y, rng.choice(labels, size=len(y), p=props.to_numpy()))

    # 3) Uniform random
    rows["uniform random"] = compute_metrics(
        y, rng.choice(labels, size=len(y)))

    out = pd.DataFrame(rows).T[["accuracy", "f1_macro", "f1_weighted"]].round(4)
    out.index.name = "baseline"

    log.info("Strongest trivial baseline: accuracy=%.4f, macro-F1=%.4f",
             out["accuracy"].max(), out["f1_macro"].max())
    return out


def class_weights(y_train, labels: list[str] | None = None) -> dict[str, float]:
    """
    Class weights for the loss function (mainly for BERT).

    Formula:  weight_c = n_total / (n_classes * n_c)

    This is the logic of scikit-learn's `class_weight='balanced'`. Errors on
    the rarer class get a proportionally larger penalty, so the model cannot
    settle for predicting the majority.

    For the classical models this is already set (`class_weight='balanced'`
    in models_classical.py). For the transformer it has to be passed to the
    loss function MANUALLY — see the WeightedTrainer class in notebook 03.
    """
    labels = labels or list(CFG.labels)
    y = pd.Series(np.asarray(y_train).astype(str))
    counts = y.value_counts().reindex(labels, fill_value=0)
    n, k = len(y), (counts > 0).sum()
    weights = {c: (n / (k * m) if m > 0 else 0.0) for c, m in counts.items()}
    log.info("Class weights: %s", {k2: round(v, 3) for k2, v in weights.items()})
    return weights


def per_class_report(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Per-class precision/recall/F1 table — can go straight into the thesis."""
    labels = labels or list(CFG.labels)
    rep = classification_report(
        np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str),
        labels=labels, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T


# ==========================================================================
# 2) CONFUSION MATRIX
# ==========================================================================
def plot_confusion_matrix(
    y_true, y_pred,
    labels: list[str] | None = None,
    title: str = "Confusion matrix",
    normalize: bool = True,
    save_as: str | None = None,
    ax=None,
):
    """
    Normalised confusion matrix (per row = what percentage of the given TRUE
    class the model predicted as which class).

    Watch for this in the thesis: with 3-class sentiment you will almost
    always see the neutral class "bleeding" into the other two. This will be
    your most interesting observation — and the basis of the error analysis
    chapter.
    """
    import matplotlib.pyplot as plt

    labels = labels or list(CFG.labels)
    cm = confusion_matrix(
        np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=labels
    )
    cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.2, 4.4))

    im = ax.imshow(cm_show, cmap="Blues", vmin=0, vmax=cm_show.max())
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    thresh = cm_show.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{cm_show[i, j]:.1%}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if cm_show[i, j] > thresh else "black")

    if created:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        if save_as:
            save_figure(save_as)
    return ax


def save_figure(name: str, dpi: int = 300) -> Path:
    """
    Save a figure to the results/figures/ folder at 300 DPI.

    300 DPI is print quality — it will not look pixelated in the thesis.
    Save EVERY figure you want to use; re-running the notebook three months
    later is much more painful.
    """
    import matplotlib.pyplot as plt
    PATHS.ensure()
    path = PATHS.figures / (name if name.endswith((".png", ".pdf")) else f"{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Figure saved: %s", path)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save a table to CSV so it can be copied into the thesis."""
    PATHS.ensure()
    path = PATHS.tables / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, encoding="utf-8")
    log.info("Table saved: %s", path)
    return path


# ==========================================================================
# 3) ERROR ANALYSIS
# ==========================================================================
def error_analysis(
    df: pd.DataFrame,
    y_pred,
    text_col: str = "text",
    label_col: str = "sentiment",
    n_examples: int = 10,
    seed: int = CFG.random_seed,
) -> pd.DataFrame:
    """
    Collect the misclassified reviews, grouped by error type.

    Chapter 8 of the thesis explicitly asks for an analysis of the
    "misclassified reviews". Run this, read 20-30 examples by hand, and write
    a few paragraphs about them — this kind of qualitative analysis is worth
    far more to the reviewer than yet another table.
    """
    out = df.copy()
    out["predicted"] = np.asarray(y_pred).astype(str)
    out["true"] = out[label_col].astype(str)
    errors = out[out["predicted"] != out["true"]].copy()
    errors["error_type"] = errors["true"] + " -> " + errors["predicted"]

    log.info("Total errors: %d / %d (%.1f%%)",
             len(errors), len(out), 100 * len(errors) / max(len(out), 1))
    log.info("Most frequent error types:\n%s",
             errors["error_type"].value_counts().to_string())

    # At most n_examples random examples per error type.
    #
    # NOT with groupby().apply(): from pandas 3.0 apply no longer passes on
    # the grouping column, so "error_type" would silently disappear. The
    # solution: first shuffle the rows randomly (with the seed, reproducibly),
    # then take the first n per group. This keeps every column and also
    # handles small groups (fewer than n errors) correctly.
    samples = (errors.sample(frac=1.0, random_state=seed)
                     .groupby("error_type", sort=False)
                     .head(n_examples)
                     .sort_values("error_type", kind="stable"))
    cols = ["error_type", "true", "predicted", "score", text_col]
    return samples[[c for c in cols if c in samples.columns]].reset_index(drop=True)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Arrange the metrics of several models into one table — this will be the
    main comparison table of the thesis (RQ1).

        compare_models({
            "Naive Bayes":  metrics_nb,
            "Logistic Reg": metrics_lr,
            "Linear SVM":   metrics_svm,
            "BERT":         metrics_bert,
        })
    """
    return (pd.DataFrame(results).T
            [["accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted"]]
            .sort_values("f1_macro", ascending=False)
            .round(4))
