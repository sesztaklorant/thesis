"""Metrics, confusion matrices, error examples and saving of tables and figures.

All models are evaluated with these functions. The main metric is macro-F1, which
weights every class equally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .config import CFG, PATHS

log = logging.getLogger(__name__)


# ==========================================================================
# 1) METRICS
# ==========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro and weighted precision/recall/F1."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                             zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                          zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f_mac,
        "precision_weighted": p_wt, "recall_weighted": r_wt, "f1_weighted": f_wt,
    }


def baseline_metrics(y_true, labels: list[str] | None = None) -> pd.DataFrame:
    """
    Trivial baselines (always the majority class, stratified and uniform
    random) - every model has to beat these.
    """
    labels = labels or list(CFG.labels)
    y = np.asarray(y_true).astype(str)
    rng = np.random.default_rng(CFG.random_seed)
    props = pd.Series(y).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    majority = props.idxmax()
    rows = {
        f"majority class ('{majority}')": compute_metrics(y, np.full(len(y), majority)),
        "stratified random": compute_metrics(y, rng.choice(labels, size=len(y), p=props.to_numpy())),
        "uniform random": compute_metrics(y, rng.choice(labels, size=len(y))),
    }
    out = pd.DataFrame(rows).T[["accuracy", "f1_macro", "f1_weighted"]].round(4)
    out.index.name = "baseline"
    return out


def per_class_report(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Per-class precision/recall/F1."""
    labels = labels or list(CFG.labels)
    rep = classification_report(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str),
                                labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Metrics of several models in one table."""
    return (pd.DataFrame(results).T
            [["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]]
            .sort_values("f1_macro", ascending=False).round(4))


# ==========================================================================
# 2) CONFUSION MATRIX
# ==========================================================================
def plot_confusion_matrix(y_true, y_pred, labels: list[str] | None = None,
                          title: str = "Confusion matrix", normalize: bool = True,
                          save_as: str | None = None, ax=None):
    """
    Row-normalised confusion matrix (what % of the given true class the model
    predicted as which class).
    """
    import matplotlib.pyplot as plt

    labels = labels or list(CFG.labels)
    cm = confusion_matrix(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=labels)
    shown = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(shown.max(), 1e-9))
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.grid(visible=False)
    threshold = shown.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{shown[i, j]:.1%}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > threshold else "black")
    if created:
        plt.tight_layout()
        if save_as:
            save_figure(save_as)
    return ax


# ==========================================================================
# 3) SAVING
# ==========================================================================
def save_figure(name: str, dpi: int = 300) -> Path:
    """Save a figure at 300 DPI to results/figures/."""
    import matplotlib.pyplot as plt
    PATHS.ensure()
    path = PATHS.figures / (name if name.endswith((".png", ".pdf")) else f"{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Figure saved: %s", path.name)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save a table as CSV to results/tables/."""
    PATHS.ensure()
    path = PATHS.tables / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, encoding="utf-8")
    log.info("Table saved: %s", path.name)
    return path


# ==========================================================================
# 4) ERROR ANALYSIS
# ==========================================================================
def error_analysis(df: pd.DataFrame, y_pred, text_col: str = "text", label_col: str = "label_final",
                   n_examples: int = 10, seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    Misclassified reviews per error type (at most n_examples each).
    Shuffle + groupby().head() - not groupby().apply(), because from pandas 3.0
    apply drops the grouping column.
    """
    out = df.copy()
    out["predicted"] = np.asarray(y_pred).astype(str)
    out["true"] = out[label_col].astype(str)
    errors = out[out["predicted"] != out["true"]].copy()
    errors["error_type"] = errors["true"] + " -> " + errors["predicted"]
    log.info("Errors: %d / %d (%.1f%%)", len(errors), len(out), 100 * len(errors) / max(len(out), 1))
    samples = (errors.sample(frac=1.0, random_state=seed)
                     .groupby("error_type", sort=False).head(n_examples)
                     .sort_values("error_type", kind="stable"))
    cols = ["error_type", "true", "predicted", "score", text_col]
    return samples[[c for c in cols if c in samples.columns]].reset_index(drop=True)
