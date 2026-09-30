"""
descriptives.py — Descriptive statistics and crosstabs on the cleaned data.

Chapter 9 of the thesis (Exploratory Data Analysis).

What is this module for?
------------------------
After the EDA comes the question: "what can I say about the data beyond how
many rows it has?" The metadata of the Booking.com dataset (country,
accommodation type, guest type, star rating) make it possible to examine
sentiment ALONG these dimensions. This is what an analyst of a hotel chain
would do too — and exactly why it is interesting in the thesis.

The central measure: LIFT
-------------------------
It is not enough to say "35% of hostels are negative". You need to know
whether that is a lot or a little. Lift tells you:

    lift = negative share of the group / negative share of the whole dataset

    lift = 1.0  ->  exactly average
    lift = 1.4  ->  negative 40% more often than average
    lift = 0.7  ->  30% less often

This measure makes groups of very different sizes comparable, and tells in
a single number where the problem is.

Figures
-------
Every figure uses the CFG colour palette (red-blue diverging, not
red-green — see the reasoning about colour blindness in config.py).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CFG, INK, SENTIMENT_COLORS, DIVERGING_BLUE_RED, sentiment_palette
from .evaluation import save_figure, save_table

log = logging.getLogger(__name__)


def _fmt_cat(v) -> str:
    """
    Category name for plotting. Converts floating-point whole numbers
    (3.0, 4.0) to integers, because a "3.0 stars" label is distracting.
    """
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v)


def prepare_star_rating(df: pd.DataFrame,
                        col: str = "accommodation_star_rating") -> pd.DataFrame:
    """
    A star rating of 0 is NOT "zero stars" but "no rating".

    If this is not handled, 0 appears as a category of its own in the
    crosstabs and looks like the worst rating when sorting — while it is
    actually missing data. In your sample this is tens of thousands of rows.

    This function turns it into a separate "n/a" category, so the analysis by
    rating covers only the properties that actually have a rating, and the
    rest is visibly kept apart.
    """
    if col not in df.columns:
        return df
    out = df.copy()
    n_zero = int((pd.to_numeric(out[col], errors="coerce") == 0).sum())
    out[col] = pd.to_numeric(out[col], errors="coerce").map(
        lambda v: "n/a" if (pd.isna(v) or v == 0) else _fmt_cat(v)
    )
    if n_zero:
        log.info("Star rating: %d rows (%.1f%%) had the value 0 -> moved to the "
                 "'n/a' category.", n_zero, 100 * n_zero / len(out))
    return out


# ==========================================================================
# 1) OVERVIEW TABLE
# ==========================================================================
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-table summary of the dataset — this goes into chapter 4 of the thesis
    ("Data Sources") and at the beginning of chapter 9.
    """
    rows: list[tuple[str, object]] = [
        ("Number of reviews", f"{len(df):,}"),
    ]

    if "accommodation_id" in df.columns:
        rows.append(("Unique properties", f"{df['accommodation_id'].nunique():,}"))
    if "accommodation_country" in df.columns:
        rows.append(("Number of countries", f"{df['accommodation_country'].nunique():,}"))
    if "accommodation_type" in df.columns:
        rows.append(("Accommodation types", f"{df['accommodation_type'].nunique():,}"))

    rows += [
        ("Mean score", f"{df['score'].mean():.2f}"),
        ("Median score", f"{df['score'].median():.2f}"),
        ("Score standard deviation", f"{df['score'].std():.2f}"),
    ]

    if "n_words" in df.columns:
        rows += [
            ("Mean review length (words)", f"{df['n_words'].mean():.1f}"),
            ("Median review length (words)", f"{df['n_words'].median():.0f}"),
        ]

    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts(normalize=True)
        for label in CFG.labels:
            rows.append((f"Share of {label}", f"{vc.get(label, 0):.1%}"))

    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column. Returns an empty table if nothing is missing."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame(columns=["missing", "share"])
    return pd.DataFrame({"missing": miss, "share": (miss / len(df)).round(4)})


def numeric_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Descriptive statistics of the numeric columns, with percentiles."""
    cols = cols or [c for c in ["score", "n_words", "n_chars", "room_nights",
                                "accommodation_score", "accommodation_star_rating",
                                "review_helpful_votes"]
                    if c in df.columns]
    return (df[cols]
            .describe(percentiles=[.05, .25, .5, .75, .95])
            .T.round(2))


# ==========================================================================
# 2) CROSSTAB: SENTIMENT ALONG ONE DIMENSION
# ==========================================================================
def sentiment_by(
    df: pd.DataFrame,
    dim: str,
    min_count: int = 200,
    top_n: int | None = 15,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Sentiment distribution along a categorical dimension, with lift values.

    Parameters
    ----------
    dim       : name of the column (e.g. "accommodation_country", "guest_type")
    min_count : categories with fewer reviews than this are dropped.
                WHY? Because a 50% negative share computed from 12 reviews is
                statistical noise, not a finding. This threshold has to be
                mentioned in the thesis.
    top_n     : only the N categories with the most reviews are kept.

    Returns
    -------
    Index: the categories. Columns:
        n           — number of reviews
        share       — what percentage of the whole dataset
        negative / neutral / positive — sentiment distribution within the group
        neg_lift    — how many times the average negative share
        mean_score  — mean guest score
    """
    if dim not in df.columns:
        raise KeyError(f"No such column: {dim}. Available: {list(df.columns)}")

    sent = df[sentiment_col].astype(str)
    base_neg = (sent == "negative").mean()
    base_pos = (sent == "positive").mean()

    g = df.assign(_s=sent).groupby(dim, observed=True)

    tab = pd.DataFrame({"n": g.size()})
    shares = (g["_s"].value_counts(normalize=True)
                .unstack(fill_value=0.0)
                .reindex(columns=list(CFG.labels), fill_value=0.0))
    tab = tab.join(shares)

    tab["neg_lift"] = tab["negative"] / base_neg if base_neg else np.nan
    tab["pos_lift"] = tab["positive"] / base_pos if base_pos else np.nan
    tab["mean_score"] = g["score"].mean()
    tab["share"] = tab["n"] / len(df)

    tab = tab[tab["n"] >= min_count]
    if top_n:
        tab = tab.nlargest(top_n, "n")

    cols = ["n", "share", *CFG.labels, "neg_lift", "pos_lift", "mean_score"]
    return tab[cols].sort_values("neg_lift", ascending=False).round(4)


def plot_sentiment_by(
    table: pd.DataFrame,
    title: str,
    xlabel: str = "Sentiment distribution within the group",
    save_as: str | None = None,
    show_n: bool = True,
    figsize: tuple[float, float] = (9, None),
):
    """
    Horizontal bar chart stacked to 100%.

    Why THIS form?
      - The question is "part-to-whole proportion across several categories"
        -> stacked bar.
      - Horizontal, because the category names (country names, accommodation
        types) are long; vertically they would overlap.
      - Normalised to 100%, because the group sizes differ a lot — raw counts
        would drown the small ones. The count appears as text at the end of
        the bar.
      - A 2px surface-coloured gap between segments: this way the boundary is
        visible even if two adjacent colours have a similar tone (e.g. in print).
    """
    import matplotlib.pyplot as plt

    data = table[list(CFG.labels)].sort_values("negative", ascending=True)

    # The axis is CATEGORICAL. If the index is numeric (e.g. star rating:
    # 0.0, 2.0, 3.0, 3.5 ...), matplotlib positions the bars on the axis AS
    # NUMBERS — then they overlap and the labels shift. Converted to strings,
    # every category gets an equal bar.
    ypos = [_fmt_cat(i) for i in data.index]

    # The height comes from the number of categories, but with a lower bound:
    # the title and the legend above it need space, otherwise tight_layout
    # complains.
    h = figsize[1] or max(3.4, 0.42 * len(data) + 2.2)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    left = pd.Series(0.0, index=data.index)
    for label in CFG.labels:
        ax.barh(
            ypos, data[label], left=left.values, label=label,
            color=SENTIMENT_COLORS[label], height=0.72,
            edgecolor=INK["surface"], linewidth=1.5,   # the 2px gap
        )
        left += data[label]

    # Direct label only on the negative segment — this is what we want to
    # read. Writing a number on every segment would be cluttered and unreadable.
    for y, (_, row) in enumerate(data.iterrows()):
        v = row["negative"]
        if v > 0.06:
            ax.text(v / 2, y, f"{v:.0%}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="medium")

    if show_n and "n" in table.columns:
        ns = table.loc[data.index, "n"]
        for y, n in enumerate(ns):
            ax.text(1.012, y, f"n={n:,}", va="center", ha="left",
                    fontsize=8.5, color=INK["muted"])

    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    # The legend goes BELOW the title, not below the chart. This way it is not
    # squeezed out even with few categories (tight_layout would warn otherwise).
    ax.set_title(title, loc="left", pad=34, color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) HEATMAP: TWO DIMENSIONS TOGETHER
# ==========================================================================
def negativity_heatmap(
    df: pd.DataFrame,
    rows: str,
    cols: str,
    min_count: int = 100,
    top_rows: int = 10,
    top_cols: int = 8,
    sentiment_col: str = "sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Crosstab of two dimensions: what is the negative share in each cell?

    Returns: (lift table, count table, average negative share)

    The lift table shows how many times the negative share of the given cell
    is the average of the whole dataset. This way cells of very different
    sizes are comparable too.
    """
    sent = df[sentiment_col].astype(str)
    base_neg = (sent == "negative").mean()

    work = df.assign(_neg=(sent == "negative").astype(float))

    keep_r = work[rows].value_counts().nlargest(top_rows).index
    keep_c = work[cols].value_counts().nlargest(top_cols).index
    work = work[work[rows].isin(keep_r) & work[cols].isin(keep_c)]

    counts = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="size")
    neg_share = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="mean")

    # Unreliably small cells are masked out
    neg_share = neg_share.where(counts >= min_count)
    lift = neg_share / base_neg if base_neg else neg_share

    # Completely empty rows/columns would only take up space in the figure
    lift = lift.dropna(how="all", axis=0).dropna(how="all", axis=1)
    counts = counts.reindex(index=lift.index, columns=lift.columns)

    if lift.empty:
        raise ValueError(
            f"Every cell is masked out (min_count={min_count}). "
            "Lower min_count or choose fewer categories "
            "(top_rows / top_cols)."
        )

    return lift.round(3), counts, base_neg


def plot_negativity_heatmap(
    lift: pd.DataFrame,
    counts: pd.DataFrame,
    base_neg: float,
    title: str,
    save_as: str | None = None,
    annotate_n: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """
    Diverging heatmap: blue = FEWER negatives than average, red = MORE.

    Why diverging and not sequential?
      Because the question is not "how much?" but "better or worse than
      average?" — that is a polarity with a meaningful midpoint (lift = 1.0).
      On a sequential scale you could not see where the average is.

    Grey cells: too few reviews, not interpretable. This has to be marked in
    the thesis too, it must not be left blank without an explanation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div_br", DIVERGING_BLUE_RED)
    cmap.set_bad(INK["grid"])

    vals = lift.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("Every cell is masked out — lower min_count.")

    # The scale is symmetric around lift = 1.0 (the average), BUT cut off at
    # 0 from below: a ratio cannot be negative, and a colour scale reaching
    # below 0 would be uninterpretable for the reader.
    span = max(abs(finite - 1).max(), 0.15)
    norm = TwoSlopeNorm(vmin=max(0.0, 1 - span), vcenter=1.0, vmax=1 + span)

    fig, ax = plt.subplots(figsize=(1.05 * lift.shape[1] + 3.4,
                                    0.56 * lift.shape[0] + 2.6))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(lift.shape[1]), [_fmt_cat(c) for c in lift.columns],
                  rotation=30, ha="right")
    ax.set_yticks(range(lift.shape[0]), [_fmt_cat(i) for i in lift.index])
    ax.set_xlabel(xlabel if xlabel is not None else (lift.columns.name or ""))
    ax.set_ylabel(ylabel if ylabel is not None else (lift.index.name or ""))
    ax.set_title(title, loc="left", pad=12, color=INK["primary"])
    ax.grid(visible=False)

    for i in range(lift.shape[0]):
        for j in range(lift.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center",
                        color=INK["muted"], fontsize=9)
                continue
            # White text on a dark cell, black on a light one
            dark = abs(v - 1) > span * 0.55
            txt = f"{v:.2f}"
            if annotate_n:
                txt += f"\nn={int(counts.iloc[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK["primary"])

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(f"Negative share / average ({base_neg:.1%})", fontsize=9.5)
    cb.outline.set_visible(False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 4) SIMPLE FREQUENCY CHART
# ==========================================================================
def plot_top_categories(
    df: pd.DataFrame,
    col: str,
    top_n: int = 12,
    title: str | None = None,
    save_as: str | None = None,
):
    """
    Simple horizontal bar chart: which category has the most reviews?

    One data series -> NO legend (the title names it), and the numbers sit
    directly at the end of the bars.
    """
    import matplotlib.pyplot as plt

    vc = df[col].value_counts().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.38 * len(vc) + 1.2)))

    ax.barh(vc.index.astype(str), vc.values, color="#2a78d6", height=0.72)
    for y, v in enumerate(vc.values):
        ax.text(v * 1.01, y, f"{v:,}", va="center", fontsize=9,
                color=INK["secondary"])

    ax.set_xlabel("Number of reviews")
    ax.set_title(title or f"Most frequent values — {col}", loc="left",
                 pad=12, color=INK["primary"])
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 5) SUMMARY RUN
# ==========================================================================
def run_all_crosstabs(
    df: pd.DataFrame,
    dims: dict[str, str] | None = None,
    min_count: int = 200,
    top_n: int = 12,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run the crosstabs on every available dimension and save the tables and
    the figures.

    dims: {column name: "readable name for the figure title"}
          If None, the default set, as far as it is present in the data.

    Returns: {column name: table}
    """
    import matplotlib.pyplot as plt

    default_dims = {
        "accommodation_country": "accommodation country",
        "accommodation_type": "accommodation type",
        "guest_type": "guest type",
        "accommodation_star_rating": "star rating",
        "guest_country": "guest country",
    }
    dims = dims or {k: v for k, v in default_dims.items() if k in df.columns}

    results: dict[str, pd.DataFrame] = {}
    for col, label in dims.items():
        if col not in df.columns:
            log.warning("Skipped (no such column): %s", col)
            continue
        try:
            tab = sentiment_by(df, col, min_count=min_count, top_n=top_n)
        except Exception as e:                       # noqa: BLE001
            log.warning("Skipped (%s): %s", col, e)
            continue
        if tab.empty:
            log.warning("Skipped (empty after min_count=%d filtering): %s", min_count, col)
            continue

        results[col] = tab
        if save:
            save_table(tab, f"01_crosstab_{col}")
            plot_sentiment_by(tab, f"Sentiment distribution — {label}",
                              save_as=f"01_crosstab_{col}")
            plt.close("all")
        log.info("%-28s %d categories, neg_lift range %.2f–%.2f",
                 col, len(tab), tab["neg_lift"].min(), tab["neg_lift"].max())

    return results
