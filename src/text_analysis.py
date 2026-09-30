"""
text_analysis.py — Distinctive terms and negation analysis.

Answers the question: "when should words be put together to get their
meaning — 'good' vs 'not good'?"

The problem with plain frequency
--------------------------------
The first version of notebook 01 showed the MOST FREQUENT words per class.
The result was almost useless: the same words led in all three classes
(room, location, staff, good). This is not a bug but a limitation of the
method — frequent words are frequent everywhere.

What we want to know is a different question:

    NOT: "which words are frequent in negative reviews?"
    BUT: "which words are more frequent in the negative class THAN IN THE OTHERS?"

This is the question of distinctive terms, and it needs a separate measure.

The method used: weighted log-odds ratio
----------------------------------------
The "Fightin' Words" method of Monroe, Colaresi and Quinn (2008), a standard
tool of political text analysis that also works well for sentiment. It has
two steps:

1. LOG-ODDS: how much more likely is the word in one class than in the
   other? The logarithm makes it symmetric (twice as frequent and half as
   frequent end up at the same distance from zero).

2. NORMALISING TO A Z-SCORE: the log-odds of rare words is uncertain — the
   ratio of a word occurring 3 times can spike by chance. Dividing by the
   standard deviation of the estimate pushes the rare words back, and the top
   of the list keeps expressions that are frequent AND distinctive.

Reference for the thesis:
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words:
    Lexical Feature Selection and Evaluation for Identifying the Content of
    Political Conflict. Political Analysis, 16(4), 372–403.
    https://doi.org/10.1093/pan/mpn018
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
def distinctive_terms(
    texts: pd.Series,
    labels: pd.Series,
    ngram_range: tuple[int, int] = (1, 2),
    top_n: int = 20,
    min_count: int = 30,
    alpha: float = 0.01,
    max_features: int = 60_000,
) -> pd.DataFrame:
    """
    Return the most characteristic terms per class.

    Parameters
    ----------
    ngram_range : (1,1) words only, (1,2) words + word pairs, (1,3) triples too.
        The WORD PAIRS are what matters: "not clean" exists only as a bigram.
    min_count   : rarer terms are dropped (noise).
    alpha       : smoothing parameter. Prevents division by zero and dampens
        very rare terms.

    Returns
    -------
    Long-format table: sentiment | rank | term | z_score |
    count_in_class | count_elsewhere | ratio
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(ngram_range=ngram_range, min_df=min_count,
                         max_features=max_features)
    X = cv.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(cv.get_feature_names_out())
    labels = labels.astype(str).to_numpy()

    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    grand_total = total.sum()
    V = len(vocab)

    rows = []
    for cls in CFG.labels:
        mask = labels == cls
        if mask.sum() == 0:
            continue

        y = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)   # within the class
        rest = total - y                                            # the other classes

        n_y, n_rest = y.sum(), rest.sum()

        # Smoothed proportions
        p_y = (y + alpha) / (n_y + alpha * V)
        p_r = (rest + alpha) / (n_rest + alpha * V)

        # Log-odds ratio
        lo = np.log(p_y / (1 - p_y)) - np.log(p_r / (1 - p_r))

        # Standard deviation of the estimate, and the z-score derived from it
        var = 1.0 / (y + alpha) + 1.0 / (rest + alpha)
        z = lo / np.sqrt(var)

        # Only the terms that actually occur are of interest
        ok = y >= max(min_count // 3, 3)
        idx = np.argsort(np.where(ok, z, -np.inf))[-top_n:][::-1]

        for rank, i in enumerate(idx, start=1):
            if not np.isfinite(z[i]):
                continue
            rows.append({
                "sentiment": cls,
                "rank": rank,
                "term": vocab[i],
                "z_score": float(z[i]),
                "count_in_class": int(y[i]),
                "count_elsewhere": int(rest[i]),
                "ratio": float((y[i] / max(n_y, 1)) / max(rest[i] / max(n_rest, 1), 1e-12)),
            })

    log.info("Vocabulary: %d terms (%s-gram), %d tokens in total",
             V, "-".join(map(str, ngram_range)), int(grand_total))
    return pd.DataFrame(rows)


def plot_distinctive_terms(
    table: pd.DataFrame,
    top_n: int = 15,
    save_as: str | None = None,
):
    """
    Horizontal bar chart of the z-scores per class.

    One data series per panel -> no legend, the panel title names it.
    """
    import matplotlib.pyplot as plt

    classes = [c for c in CFG.labels if c in set(table["sentiment"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 5.6))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = (table[table["sentiment"] == cls]
               .nsmallest(top_n, "rank")
               .sort_values("z_score"))
        ax.barh(sub["term"], sub["z_score"],
                color=SENTIMENT_COLORS[cls], height=0.72)
        ax.set_title(cls, loc="left", pad=10, color=INK["primary"])
        ax.set_xlabel("z-score (distinctiveness)")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.margins(x=0.10)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return axes


# ==========================================================================
# 2) NEGATION ANALYSIS
# ==========================================================================
NEGATORS = ["not", "no", "never", "without", "nothing", "hardly", "barely",
            "didn't", "wasn't", "weren't", "isn't", "aren't", "couldn't",
            "wouldn't", "doesn't", "don't", "can't", "won't", "haven't", "hadn't"]


def negation_bigrams(
    texts: pd.Series,
    labels: pd.Series,
    top_n: int = 25,
    min_count: int = 20,
) -> pd.DataFrame:
    """
    Collect the word pairs STARTING WITH A NEGATION and show their
    distribution across the classes.

    This is the direct answer to your question: how many "not good",
    "not clean", "no hot water" occur, and in which class?

    Returns: bigram | total | negative/neutral/positive share | neg_lift
    """
    from sklearn.feature_extraction.text import CountVectorizer

    neg_pattern = re.compile(r"^(" + "|".join(re.escape(n) for n in NEGATORS) + r") ")

    cv = CountVectorizer(ngram_range=(2, 2), min_df=min_count)
    X = cv.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(cv.get_feature_names_out())

    keep = np.array([bool(neg_pattern.match(v)) for v in vocab])
    if not keep.any():
        log.warning("No word pair starting with a negation was found. The "
                    "negation words may already have been removed from the text "
                    "column — check that keep_negations=True was used in cleaning.")
        return pd.DataFrame()

    X = X[:, keep]
    vocab = vocab[keep]
    lab = labels.astype(str).to_numpy()

    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    base_neg = (lab == "negative").mean()

    shares = {}
    for cls in CFG.labels:
        m = lab == cls
        shares[cls] = np.asarray(X[m].sum(axis=0)).ravel().astype(float)

    df = pd.DataFrame({"bigram": vocab, "total": total.astype(int)})
    for cls in CFG.labels:
        df[cls] = shares[cls] / np.maximum(total, 1)
    df["neg_lift"] = df["negative"] / base_neg if base_neg else np.nan

    return (df.nlargest(top_n, "total")
              .set_index("bigram")
              .sort_values("neg_lift", ascending=False)
              .round(4))


def compare_with_without_negation(
    texts: pd.Series,
    labels: pd.Series,
    words: list[str] | None = None,
) -> pd.DataFrame:
    """
    The core of your question: how much does a "not" before the word matter?

    For each word it compares:
        "good"      what percentage falls into negative reviews?
        "not good"  what percentage falls into negative reviews?

    A large difference between the two is QUANTITATIVE EVIDENCE that the
    unigram-based (bag-of-words) representation loses something — which is
    exactly the point of RQ2.

    The table can go into chapter 6.2 of the thesis.
    """
    words = words or ["good", "clean", "comfortable", "friendly", "helpful",
                      "nice", "quiet", "great", "big", "working", "hot water",
                      "problem", "recommend"]

    txt = texts.fillna("").astype(str).str.lower()
    lab = labels.astype(str)
    base = (lab == "negative").mean()

    # A non-capturing group (?:...) is needed, otherwise pandas warns that the
    # pattern has a capturing group and recommends str.extract instead.
    neg_prefixes = "(?:" + "|".join(re.escape(n) for n in NEGATORS) + ")"

    rows = []
    for w in words:
        wq = re.escape(w)
        # Does the word occur at all? (with word boundaries, so that "good"
        # does not match inside "goodbye")
        present = txt.str.contains(rf"(?<!\w){wq}(?!\w)", regex=True, na=False)
        # Does it occur DIRECTLY AFTER A NEGATION?
        negated = txt.str.contains(rf"(?<!\w){neg_prefixes}\s+{wq}(?!\w)",
                                   regex=True, na=False)
        # "Plain" = present, but not negated. (If both forms occur in one
        # review, it is put into the negated group — more conservative,
        # because negation is the stronger signal.)
        plain_only = present & ~negated

        if plain_only.sum() < 20 or negated.sum() < 20:
            continue

        p_neg_plain = (lab[plain_only] == "negative").mean()
        p_neg_negated = (lab[negated] == "negative").mean()

        rows.append({
            "word": w,
            "n (plain)": int(plain_only.sum()),
            "negative share (plain)": p_neg_plain,
            "n (negated)": int(negated.sum()),
            "negative share (negated)": p_neg_negated,
            "difference": p_neg_negated - p_neg_plain,
            "lift (negated)": p_neg_negated / base if base else np.nan,
        })

    if not rows:
        log.warning("None of the words had enough occurrences for the comparison.")
        return pd.DataFrame()

    return (pd.DataFrame(rows)
            .set_index("word")
            .sort_values("difference", ascending=False)
            .round(4))


def plot_negation_effect(table: pd.DataFrame, save_as: str | None = None):
    """
    Paired dot plot (dumbbell): for every word the negative share of the
    plain and of the negated form, connected by a line.

    Why this form? Because the question is a PAIRED comparison — the length
    of the line shows the size of the effect directly, and the starting level
    is visible at the same time. Two separate bar charts would show the same
    thing less readably.
    """
    import matplotlib.pyplot as plt

    if table.empty:
        log.warning("Empty table, nothing to plot.")
        return None

    d = table.sort_values("difference")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.42 * len(d) + 2.0)))

    ax.hlines(y, d["negative share (plain)"], d["negative share (negated)"],
              color=INK["axis"], lw=2, zorder=1)
    ax.scatter(d["negative share (plain)"], y, s=90, zorder=2,
               color=SENTIMENT_COLORS["positive"], label="the word on its own",
               edgecolor=INK["surface"], linewidth=1.5)
    ax.scatter(d["negative share (negated)"], y, s=90, zorder=2,
               color=SENTIMENT_COLORS["negative"], label="with negation",
               edgecolor=INK["surface"], linewidth=1.5)

    ax.set_yticks(y, d.index)
    ax.set_xlabel("Share of occurrences that fall into negative reviews")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("The effect of negation — same word, different meaning",
                 loc="left", pad=34, color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) COMPARING N-GRAM LEVELS
# ==========================================================================
def ngram_level_comparison(
    texts: pd.Series,
    labels: pd.Series,
    min_count: int = 30,
) -> pd.DataFrame:
    """
    How much more "distinctive" does the vocabulary become if word pairs and
    triples are allowed?

    The table shows one n-gram level per row: how many terms there are, and
    the mean distinctiveness of the top-50 terms.

    This anticipates the ablation study of notebook 02: there the same
    question is measured in MODEL PERFORMANCE, here it is looked at on the
    level of the vocabulary.
    """
    rows = []
    for name, rng in [("unigram (1,1)", (1, 1)),
                      ("uni+bigram (1,2)", (1, 2)),
                      ("uni+bi+trigram (1,3)", (1, 3))]:
        t = distinctive_terms(texts, labels, ngram_range=rng,
                              top_n=50, min_count=min_count)
        if t.empty:
            continue
        n_multi = t["term"].str.contains(" ").sum()
        rows.append({
            "n-gram level": name,
            "top-50 mean z": t.groupby("sentiment")["z_score"].mean().mean(),
            "multi-word share in top-50": n_multi / len(t),
            "example (negative)": ", ".join(
                t[(t.sentiment == "negative") & (t["term"].str.contains(" "))]
                ["term"].head(3)
            ) or "—",
        })

    return pd.DataFrame(rows).set_index("n-gram level").round(3)
