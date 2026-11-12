"""Distinctive terms and negation analysis.

Raw frequencies show the same words in every class (room, location, staff), so terms
are ranked by the weighted log-odds ratio with an informative Dirichlet prior
(Monroe, Colaresi & Quinn, 2008). The z-score of the log-odds suppresses chance peaks
of rare terms. The label column can be any label source.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import CFG, INK, SENTIMENT_COLORS
from .evaluation import save_figure

log = logging.getLogger(__name__)


# ==========================================================================
# 1) DISTINCTIVE TERMS
# ==========================================================================
def distinctive_terms(texts: pd.Series, labels: pd.Series, ngram_range: tuple[int, int] = (1, 2),
                      top_n: int = 20, min_count: int = 30, alpha: float = 0.01,
                      max_features: int = 60_000) -> pd.DataFrame:
    """
    The most characteristic terms per class (long-format table).
    alpha: smoothing (against division by zero, damps rare terms).

    Returns: sentiment | rank | term | z_score | count_in_class | count_elsewhere | ratio
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_count, max_features=max_features)
    X = vectorizer.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = labels.astype(str).to_numpy()
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    vocab_size = len(vocab)

    rows = []
    for cls in CFG.labels:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        in_class = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)
        elsewhere = total - in_class
        n_in, n_else = in_class.sum(), elsewhere.sum()

        p_in = (in_class + alpha) / (n_in + alpha * vocab_size)
        p_else = (elsewhere + alpha) / (n_else + alpha * vocab_size)
        log_odds = np.log(p_in / (1 - p_in)) - np.log(p_else / (1 - p_else))
        z = log_odds / np.sqrt(1.0 / (in_class + alpha) + 1.0 / (elsewhere + alpha))

        ok = in_class >= max(min_count // 3, 3)
        top_idx = np.argsort(np.where(ok, z, -np.inf))[-top_n:][::-1]
        for rank, i in enumerate(top_idx, start=1):
            if not np.isfinite(z[i]):
                continue
            rows.append({
                "sentiment": cls, "rank": rank, "term": vocab[i], "z_score": float(z[i]),
                "count_in_class": int(in_class[i]), "count_elsewhere": int(elsewhere[i]),
                "ratio": float((in_class[i] / max(n_in, 1)) / max(elsewhere[i] / max(n_else, 1), 1e-12)),
            })
    log.info("Vocabulary: %d terms (%s-gram)", vocab_size, "-".join(map(str, ngram_range)))
    return pd.DataFrame(rows)


def plot_distinctive_terms(table: pd.DataFrame, top_n: int = 15, save_as: str | None = None):
    """Horizontal bar chart per class (z-score)."""
    import matplotlib.pyplot as plt

    classes = [c for c in CFG.labels if c in set(table["sentiment"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 5.6))
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        sub = table[table["sentiment"] == cls].nsmallest(top_n, "rank").sort_values("z_score")
        ax.barh(sub["term"], sub["z_score"], color=SENTIMENT_COLORS[cls], height=0.72)
        ax.set_title(cls, loc="left", pad=10, color=INK["primary"])
        ax.set_xlabel("z-score (distinctiveness)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.10)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return axes


def top_frequent_terms(texts: pd.Series, labels: pd.Series, ngram_range: tuple[int, int] = (1, 2),
                       top_n: int = 20, min_count: int = 30) -> pd.DataFrame:
    """
    The naive approach: the most frequent terms per class (plain counts, no
    weighting). Uses the same vocabulary as distinctive_terms(), so the two are
    directly comparable.

    Returns: sentiment | rank | term | count | share_in_class
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_count)
    X = vectorizer.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = labels.astype(str).to_numpy()
    rows = []
    for cls in CFG.labels:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        counts = np.asarray(X[mask].sum(axis=0)).ravel()
        total = max(counts.sum(), 1)
        for rank, i in enumerate(np.argsort(counts)[-top_n:][::-1], start=1):
            rows.append({"sentiment": cls, "rank": rank, "term": vocab[i],
                         "count": int(counts[i]), "share_in_class": counts[i] / total})
    return pd.DataFrame(rows)


def top_list_overlap(freq_table: pd.DataFrame, dist_table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Overlap of the top lists of the three classes. Nearly identical frequency lists
    say nothing about what distinguishes the classes, while the z-score lists share
    few terms.
      shared_by_all_classes - how many terms are on all three top lists
      mean_pairwise_jaccard - pairwise overlap of the lists (0 = none, 1 = identical)
    """
    import itertools

    rows = {}
    for name, table in (("most frequent (count)", freq_table), ("most distinctive (z-score)", dist_table)):
        sets = {cls: set(table[(table["sentiment"] == cls) & (table["rank"] <= top_n)]["term"])
                for cls in CFG.labels if cls in set(table["sentiment"])}
        pairs = list(itertools.combinations(sets.values(), 2))
        rows[name] = {
            f"top_n": top_n,
            "shared_by_all_classes": len(set.intersection(*sets.values())) if sets else 0,
            "mean_pairwise_jaccard": float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]))
            if pairs else float("nan"),
        }
    return pd.DataFrame(rows).T.round(3)


def plot_frequency_vs_distinctive(freq_table: pd.DataFrame, dist_table: pd.DataFrame, top_n: int = 15,
                                  min_z: float = 3.0, save_as: str | None = None):
    """
    Two-row figure on the same vocabulary:
      top row    - most frequent terms (count);
      bottom row - most distinctive terms (z-score).
    In the top row full colour marks the terms that are also on the bottom (z-score)
    top list (z ≥ min_z); pale = frequent but not characteristic (occurs everywhere).
    This shows at a glance what the weighting "gains".
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    classes = [c for c in CFG.labels if c in set(freq_table["sentiment"])]
    fig, axes = plt.subplots(2, len(classes), figsize=(5.4 * len(classes), 10.4))
    axes = np.atleast_2d(axes)

    for j, cls in enumerate(classes):
        # "Distinctive" = on the z-score top list and z ≥ min_z (truly over-represented).
        sub = dist_table[(dist_table["sentiment"] == cls) & (dist_table["rank"] <= top_n)]
        distinctive = set(sub.loc[sub["z_score"] >= min_z, "term"])

        top = freq_table[(freq_table["sentiment"] == cls) & (freq_table["rank"] <= top_n)].sort_values("count")
        # Full colour = also distinctive; pale = only frequent (frequent in every class).
        colors = [to_rgba(SENTIMENT_COLORS[cls], 1.0 if t in distinctive else 0.25) for t in top["term"]]
        ax = axes[0, j]
        ax.barh(top["term"], top["count"], color=colors, height=0.72)
        ax.set_title(f"{cls} — most frequent", loc="left", pad=8, color=INK["primary"])
        ax.set_xlabel("Occurrences in the class (count)")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.08)

        bottom = dist_table[(dist_table["sentiment"] == cls) & (dist_table["rank"] <= top_n)].sort_values("z_score")
        ax = axes[1, j]
        ax.barh(bottom["term"], bottom["z_score"], color=SENTIMENT_COLORS[cls], height=0.72)
        ax.set_title(f"{cls} — most distinctive", loc="left", pad=8, color=INK["primary"])
        ax.set_xlabel("z-score (distinctiveness)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.08)

    fig.legend(handles=[Patch(color=to_rgba(INK["secondary"], 1.0), label="full colour: also distinctive (z ≥ %g)" % min_z),
                        Patch(color=to_rgba(INK["secondary"], 0.25), label="pale: frequent, but not distinctive")],
               loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Frequency vs distinctiveness — the same vocabulary, two rankings",
                 x=0.01, y=1.035, ha="left", fontsize=13, color=INK["primary"])
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_as:
        save_figure(save_as)
    return axes


# ==========================================================================
# 2) NEGATION ANALYSIS
# ==========================================================================
NEGATORS = ["not", "no", "never", "without", "nothing", "hardly", "barely",
            "didn't", "wasn't", "weren't", "isn't", "aren't", "couldn't",
            "wouldn't", "doesn't", "don't", "can't", "won't", "haven't", "hadn't"]


def negation_bigrams(texts: pd.Series, labels: pd.Series, top_n: int = 25,
                     min_count: int = 20) -> pd.DataFrame:
    """
    Word pairs starting with a negation ("not clean", "no hot water") and their
    class distribution. neg_lift = negative share of the pair / average negative share.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    pattern = re.compile(r"^(" + "|".join(re.escape(n) for n in NEGATORS) + r") ")
    vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=min_count)
    X = vectorizer.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())
    keep = np.array([bool(pattern.match(v)) for v in vocab])
    if not keep.any():
        log.warning("No negation bigram found — were negations kept in cleaning (keep_negations=True)?")
        return pd.DataFrame()

    X, vocab = X[:, keep], vocab[keep]
    lab = labels.astype(str).to_numpy()
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    base_neg = (lab == "negative").mean()

    out = pd.DataFrame({"bigram": vocab, "total": total.astype(int)})
    for cls in CFG.labels:
        out[cls] = np.asarray(X[lab == cls].sum(axis=0)).ravel() / np.maximum(total, 1)
    out["neg_lift"] = out["negative"] / base_neg if base_neg else np.nan
    return (out.nlargest(top_n, "total").set_index("bigram")
               .sort_values("neg_lift", ascending=False).round(4))


def compare_with_without_negation(texts: pd.Series, labels: pd.Series,
                                  words: list[str] | None = None) -> pd.DataFrame:
    """
    Share of negative reviews among the occurrences of a word on its own and negated.
    A large difference shows that the single-word (unigram) representation loses
    information.
    """
    words = words or ["good", "clean", "comfortable", "friendly", "helpful", "nice",
                      "quiet", "great", "big", "working", "hot water", "problem", "recommend"]
    txt = texts.fillna("").astype(str).str.lower()
    lab = labels.astype(str)
    base = (lab == "negative").mean()
    negator_group = "(?:" + "|".join(re.escape(n) for n in NEGATORS) + ")"

    rows = []
    for w in words:
        wq = re.escape(w)
        present = txt.str.contains(rf"(?<!\w){wq}(?!\w)", regex=True, na=False)
        negated = txt.str.contains(rf"(?<!\w){negator_group}\s+{wq}(?!\w)", regex=True, na=False)
        plain_only = present & ~negated
        if plain_only.sum() < 20 or negated.sum() < 20:
            continue
        p_plain = (lab[plain_only] == "negative").mean()
        p_negated = (lab[negated] == "negative").mean()
        rows.append({"word": w, "n (plain)": int(plain_only.sum()), "negative share (plain)": p_plain,
                     "n (negated)": int(negated.sum()), "negative share (negated)": p_negated,
                     "difference": p_negated - p_plain,
                     "lift (negated)": p_negated / base if base else np.nan})
    if not rows:
        log.warning("None of the words had enough occurrences.")
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("word").sort_values("difference", ascending=False).round(4)


def plot_negation_effect(table: pd.DataFrame, save_as: str | None = None):
    """Dumbbell plot: word on its own vs. negated."""
    import matplotlib.pyplot as plt

    if table.empty:
        return None
    d = table.sort_values("difference")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.42 * len(d) + 2.0)))
    ax.hlines(y, d["negative share (plain)"], d["negative share (negated)"], color=INK["axis"], lw=2)
    ax.scatter(d["negative share (plain)"], y, s=90, zorder=2, color=SENTIMENT_COLORS["positive"],
               label="the word on its own", edgecolor=INK["surface"], linewidth=1.5)
    ax.scatter(d["negative share (negated)"], y, s=90, zorder=2, color=SENTIMENT_COLORS["negative"],
               label="with negation", edgecolor=INK["surface"], linewidth=1.5)
    ax.set_yticks(y, d.index)
    ax.set_xlabel("Share of occurrences in negative reviews")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_title("The effect of negation — same word, different meaning", loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) N-GRAM SZINTEK / N-GRAM LEVELS
# ==========================================================================
def ngram_level_comparison(texts: pd.Series, labels: pd.Series, min_count: int = 30) -> pd.DataFrame:
    """
    Distinctiveness of the vocabulary with single words, pairs and triples.
    """
    rows = []
    for name, rng in [("unigram (1,1)", (1, 1)), ("uni+bigram (1,2)", (1, 2)),
                      ("uni+bi+trigram (1,3)", (1, 3))]:
        t = distinctive_terms(texts, labels, ngram_range=rng, top_n=50, min_count=min_count)
        if t.empty:
            continue
        rows.append({
            "n-gram level": name,
            "top-50 mean z": t.groupby("sentiment")["z_score"].mean().mean(),
            "multi-word share in top-50": t["term"].str.contains(" ").mean(),
            "example (negative)": ", ".join(
                t[(t.sentiment == "negative") & t["term"].str.contains(" ")]["term"].head(3)) or "—",
        })
    return pd.DataFrame(rows).set_index("n-gram level").round(3)
