"""
text_analysis.py — Megkülönböztető kifejezések és tagadás-elemzés.
                   Distinctive terms and negation analysis.

HU: A puszta gyakoriság használhatatlan (minden osztályban ugyanazok vezetnek:
    room, location, staff). A kérdés az, hogy egy kifejezés GYAKORIBB-E az adott
    osztályban, MINT A TÖBBIBEN. Erre a súlyozott log-odds arányt használjuk
    (Monroe, Colaresi & Quinn, 2008, "Fightin' Words"):
      1. log-odds: hányszor valószínűbb a kifejezés ebben az osztályban;
      2. z-pontszám: a becslés szórásával osztva a ritka kifejezések véletlen
         kiugrásait visszanyomjuk.
EN: Plain frequency is useless (the same words lead in every class: room,
    location, staff). The question is whether a term is MORE frequent in the
    class THAN IN THE OTHERS. We use the weighted log-odds ratio
    (Monroe, Colaresi & Quinn, 2008, "Fightin' Words"):
      1. log-odds: how many times more likely the term is in this class;
      2. z-score: dividing by the standard deviation of the estimate suppresses
         chance spikes of rare terms.

HU: A címkeoszlop tetszőleges lehet (label_final, A_label, ...), így ugyanaz az
    elemzés lefuttatható bármelyik címkeforrásra.
EN: The label column can be anything (label_final, A_label, ...), so the same
    analysis can be run for any label source.

Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature
selection and evaluation for identifying the content of political conflict.
Political Analysis, 16(4), 372–403. https://doi.org/10.1093/pan/mpn018
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
# 1) MEGKÜLÖNBÖZTETŐ KIFEJEZÉSEK / DISTINCTIVE TERMS
# ==========================================================================
def distinctive_terms(texts: pd.Series, labels: pd.Series, ngram_range: tuple[int, int] = (1, 2),
                      top_n: int = 20, min_count: int = 30, alpha: float = 0.01,
                      max_features: int = 60_000) -> pd.DataFrame:
    """
    HU: Osztályonként a legjellemzőbb kifejezések (hosszú formátumú tábla).
        alpha: simítás (nullával osztás ellen, ritka kifejezések tompítása).
    EN: The most characteristic terms per class (long-format table).
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
    """HU: Vízszintes oszlopdiagram osztályonként (z-pontszám). / EN: Horizontal bar chart per class (z-score)."""
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


# ==========================================================================
# 2) TAGADÁS-ELEMZÉS / NEGATION ANALYSIS
# ==========================================================================
NEGATORS = ["not", "no", "never", "without", "nothing", "hardly", "barely",
            "didn't", "wasn't", "weren't", "isn't", "aren't", "couldn't",
            "wouldn't", "doesn't", "don't", "can't", "won't", "haven't", "hadn't"]


def negation_bigrams(texts: pd.Series, labels: pd.Series, top_n: int = 25,
                     min_count: int = 20) -> pd.DataFrame:
    """
    HU: Tagadással kezdődő szópárok ("not clean", "no hot water") és osztályeloszlásuk.
        neg_lift = a szópár negatív aránya / az átlagos negatív arány.
    EN: Word pairs starting with a negation ("not clean", "no hot water") and their
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
    HU: Ugyanaz a szó önmagában vs. tagadva: az előfordulások hány %-a esik negatív
        véleménybe? A nagy különbség számszerű bizonyíték arra, hogy az egyszavas
        (unigram) reprezentáció információt veszít.
    EN: The same word on its own vs. negated: what % of the occurrences fall into
        negative reviews? A large difference is quantitative evidence that the
        single-word (unigram) representation loses information.
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
    """HU: Páros pontdiagram: szó önmagában vs. tagadva. / EN: Dumbbell plot: word on its own vs. negated."""
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
    HU: Mennyivel "megkülönböztetőbb" a szókincs szópárokkal / hármasokkal?
    EN: How much more "distinctive" does the vocabulary become with pairs / triples?
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
