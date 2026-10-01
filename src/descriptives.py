"""
descriptives.py — Leíró statisztika és kereszttáblák. / Descriptive statistics and crosstabs.

HU: A Booking metaadatai (ország, szállástípus, vendégtípus, csillag) mentén
    vizsgáljuk a (BERT-ből származó) szentimentet. A központi mérőszám a LIFT:
        lift = a csoport negatív aránya / a teljes adat negatív aránya
        1,0 = átlagos; 1,4 = 40%-kal gyakoribb negatív; 0,7 = 30%-kal ritkább.
    Ezek ASSZOCIÁCIÓK, nem ok-okozati kapcsolatok.
EN: The (BERT-derived) sentiment is examined along the Booking metadata
    (country, accommodation type, guest type, stars). The central measure is LIFT:
        lift = negative share of the group / negative share of the whole data
        1.0 = average; 1.4 = negative 40% more often; 0.7 = 30% less often.
    These are ASSOCIATIONS, not causal relationships.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CFG, DIVERGING_BLUE_RED, INK, SENTIMENT_COLORS
from .evaluation import save_figure, save_table

log = logging.getLogger(__name__)

DEFAULT_LABEL_COL = "label_final"


def _fmt_cat(value) -> str:
    """HU: 3.0 -> "3" (a "3.0 csillag" zavaró). / EN: 3.0 -> "3" ("3.0 stars" is distracting)."""
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def prepare_star_rating(df: pd.DataFrame, col: str = "accommodation_star_rating") -> pd.DataFrame:
    """
    HU: A 0 csillag NEM "nulla csillag", hanem "nincs besorolás" -> külön "n/a" kategória.
    EN: A star rating of 0 is NOT "zero stars" but "no rating" -> separate "n/a" category.
    """
    if col not in df.columns:
        return df
    out = df.copy()
    values = pd.to_numeric(out[col], errors="coerce")
    out[col] = values.map(lambda v: "n/a" if (pd.isna(v) or v == 0) else _fmt_cat(v))
    n_zero = int((values == 0).sum())
    if n_zero:
        log.info("Star rating: %d rows (%.1f%%) had 0 -> 'n/a'.", n_zero, 100 * n_zero / len(out))
    return out


# ==========================================================================
# 1) ÁTTEKINTÉS / OVERVIEW
# ==========================================================================
def dataset_overview(df: pd.DataFrame, label_col: str | None = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """HU: Egytáblás összefoglaló az adathalmazról. / EN: One-table summary of the dataset."""
    rows: list[tuple[str, object]] = [("Number of reviews", f"{len(df):,}")]
    for col, name in [("accommodation_id", "Unique properties"),
                      ("accommodation_country", "Number of countries"),
                      ("accommodation_type", "Accommodation types")]:
        if col in df.columns:
            rows.append((name, f"{df[col].nunique():,}"))
    rows += [("Mean guest score", f"{df['score'].mean():.2f}"),
             ("Median guest score", f"{df['score'].median():.2f}"),
             ("Guest score std", f"{df['score'].std():.2f}")]
    if "n_words" in df.columns:
        rows += [("Mean review length (words)", f"{df['n_words'].mean():.1f}"),
                 ("Median review length (words)", f"{df['n_words'].median():.0f}")]
    if label_col and label_col in df.columns:
        shares = df[label_col].astype(str).value_counts(normalize=True)
        rows += [(f"Share of {label} ({label_col})", f"{shares.get(label, 0):.1%}") for label in CFG.labels]
    if "score10_final" in df.columns:
        rows.append(("Mean text-based score (1–10)", f"{df['score10_final'].mean():.2f}"))
    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """HU: Hiányzó értékek oszloponként. / EN: Missing values per column."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing": miss, "share": (miss / max(len(df), 1)).round(4)})


def numeric_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """HU: A numerikus oszlopok leíró statisztikái. / EN: Descriptive statistics of the numeric columns."""
    cols = cols or [c for c in ["score", "score10_final", "n_words", "n_chars", "room_nights",
                                "accommodation_score", "review_helpful_votes"] if c in df.columns]
    return df[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(2)


# ==========================================================================
# 2) KERESZTTÁBLA EGY DIMENZIÓ MENTÉN / CROSSTAB ALONG ONE DIMENSION
# ==========================================================================
def sentiment_by(df: pd.DataFrame, dim: str, min_count: int = 200, top_n: int | None = 15,
                 label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """
    HU: Szentiment-eloszlás egy kategoriális dimenzió mentén, lift-értékekkel.
        min_count: ennél kevesebb véleményű kategória kimarad (12 véleményből
        számolt arány zaj, nem eredmény).
    EN: Sentiment distribution along a categorical dimension, with lift values.
        min_count: categories with fewer reviews are dropped (a share computed
        from 12 reviews is noise, not a finding).
    """
    if dim not in df.columns:
        raise KeyError(f"No such column: {dim}")
    sent = df[label_col].astype(str)
    base_neg, base_pos = (sent == "negative").mean(), (sent == "positive").mean()
    grouped = df.assign(_s=sent).groupby(dim, observed=True)

    tab = pd.DataFrame({"n": grouped.size()})
    tab = tab.join(grouped["_s"].value_counts(normalize=True).unstack(fill_value=0.0)
                   .reindex(columns=list(CFG.labels), fill_value=0.0))
    tab["neg_lift"] = tab["negative"] / base_neg if base_neg else np.nan
    tab["pos_lift"] = tab["positive"] / base_pos if base_pos else np.nan
    tab["mean_guest_score"] = grouped["score"].mean()
    if "score10_final" in df.columns:
        tab["mean_text_score"] = grouped["score10_final"].mean()
    tab["share"] = tab["n"] / len(df)

    tab = tab[tab["n"] >= min_count]
    if top_n:
        tab = tab.nlargest(top_n, "n")
    cols = ["n", "share", *CFG.labels, "neg_lift", "pos_lift", "mean_guest_score"]
    cols += ["mean_text_score"] if "mean_text_score" in tab.columns else []
    return tab[cols].sort_values("neg_lift", ascending=False).round(4)


def plot_sentiment_by(table: pd.DataFrame, title: str, xlabel: str = "Sentiment distribution within the group",
                      save_as: str | None = None, show_n: bool = True):
    """HU: 100%-ra halmozott vízszintes oszlopdiagram. / EN: 100% stacked horizontal bar chart."""
    import matplotlib.pyplot as plt

    data = table[list(CFG.labels)].sort_values("negative", ascending=True)
    ypos = [_fmt_cat(i) for i in data.index]     # HU/EN: categorical axis
    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.42 * len(data) + 2.2)))
    left = np.zeros(len(data))
    for label in CFG.labels:
        ax.barh(ypos, data[label], left=left, label=label, color=SENTIMENT_COLORS[label],
                height=0.72, edgecolor=INK["surface"], linewidth=1.5)
        left += data[label].to_numpy()
    for y, v in enumerate(data["negative"]):
        if v > 0.06:
            ax.text(v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9)
    if show_n and "n" in table.columns:
        for y, n in enumerate(table.loc[data.index, "n"]):
            ax.text(1.012, y, f"n={n:,}", va="center", ha="left", fontsize=8.5, color=INK["muted"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) HŐTÉRKÉP KÉT DIMENZIÓRA / HEATMAP FOR TWO DIMENSIONS
# ==========================================================================
def negativity_heatmap(df: pd.DataFrame, rows: str, cols: str, min_count: int = 100,
                       top_rows: int = 10, top_cols: int = 8,
                       label_col: str = DEFAULT_LABEL_COL) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    HU: Két dimenzió kereszttáblája: a negatív arány liftje cellánként.
    EN: Crosstab of two dimensions: lift of the negative share per cell.

    Returns: (lift table, count table, average negative share)
    """
    sent = df[label_col].astype(str)
    base_neg = (sent == "negative").mean()
    work = df.assign(_neg=(sent == "negative").astype(float))
    work = work[work[rows].isin(work[rows].value_counts().nlargest(top_rows).index)
                & work[cols].isin(work[cols].value_counts().nlargest(top_cols).index)]
    counts = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="size")
    neg_share = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="mean")
    lift = (neg_share.where(counts >= min_count) / base_neg if base_neg else neg_share)
    lift = lift.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if lift.empty:
        raise ValueError(f"Every cell is masked out (min_count={min_count}).")
    return lift.round(3), counts.reindex(index=lift.index, columns=lift.columns), base_neg


def plot_negativity_heatmap(lift: pd.DataFrame, counts: pd.DataFrame, base_neg: float, title: str,
                            save_as: str | None = None, xlabel: str | None = None,
                            ylabel: str | None = None):
    """
    HU: Divergens hőtérkép: kék = az átlagnál KEVESEBB negatív, piros = TÖBB.
        Szürke cella: túl kevés vélemény, nem értelmezhető.
    EN: Diverging heatmap: blue = FEWER negatives than average, red = MORE.
        Grey cell: too few reviews, not interpretable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div_br", DIVERGING_BLUE_RED)
    cmap.set_bad(INK["grid"])
    vals = lift.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    span = max(abs(finite - 1).max(), 0.15)
    norm = TwoSlopeNorm(vmin=max(0.0, 1 - span), vcenter=1.0, vmax=1 + span)

    fig, ax = plt.subplots(figsize=(1.05 * lift.shape[1] + 3.4, 0.56 * lift.shape[0] + 2.6))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(lift.shape[1]), [_fmt_cat(c) for c in lift.columns], rotation=30, ha="right")
    ax.set_yticks(range(lift.shape[0]), [_fmt_cat(i) for i in lift.index])
    ax.set_xlabel(xlabel if xlabel is not None else (lift.columns.name or ""))
    ax.set_ylabel(ylabel if ylabel is not None else (lift.index.name or ""))
    ax.set_title(title, loc="left", pad=12)
    ax.grid(visible=False)
    for i in range(lift.shape[0]):
        for j in range(lift.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center", color=INK["muted"], fontsize=9)
                continue
            ax.text(j, i, f"{v:.2f}\nn={int(counts.iloc[i, j]):,}", ha="center", va="center",
                    fontsize=8.5, color="white" if abs(v - 1) > span * 0.55 else INK["primary"])
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(f"Negative share / average ({base_neg:.1%})", fontsize=9.5)
    cb.outline.set_visible(False)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def plot_top_categories(df: pd.DataFrame, col: str, top_n: int = 12, title: str | None = None,
                        save_as: str | None = None):
    """HU: Melyik kategóriában van a legtöbb vélemény? / EN: Which category has the most reviews?"""
    import matplotlib.pyplot as plt

    counts = df[col].value_counts().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.38 * len(counts) + 1.2)))
    ax.barh(counts.index.astype(str), counts.values, color="#2a78d6", height=0.72)
    for y, v in enumerate(counts.values):
        ax.text(v * 1.01, y, f"{v:,}", va="center", fontsize=9, color=INK["secondary"])
    ax.set_xlabel("Number of reviews")
    ax.set_title(title or f"Most frequent values — {col}", loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def run_all_crosstabs(df: pd.DataFrame, dims: dict[str, str] | None = None, min_count: int = 200,
                      top_n: int = 12, save: bool = True, label_col: str = DEFAULT_LABEL_COL,
                      prefix: str = "04") -> dict[str, pd.DataFrame]:
    """
    HU: Kereszttábla + ábra minden elérhető dimenzióra, automatikus mentéssel.
    EN: Crosstab + figure for every available dimension, saved automatically.
    """
    import matplotlib.pyplot as plt

    default = {"accommodation_country": "accommodation country",
               "accommodation_type": "accommodation type",
               "guest_type": "guest type",
               "accommodation_star_rating": "star rating"}
    dims = dims or {k: v for k, v in default.items() if k in df.columns}
    results = {}
    for col, name in dims.items():
        try:
            tab = sentiment_by(df, col, min_count=min_count, top_n=top_n, label_col=label_col)
        except Exception as e:                                    # noqa: BLE001
            log.warning("Skipped %s: %s", col, e)
            continue
        if tab.empty:
            continue
        results[col] = tab
        if save:
            save_table(tab, f"{prefix}_crosstab_{col}")
            plot_sentiment_by(tab, f"Sentiment distribution — {name}", save_as=f"{prefix}_crosstab_{col}")
            plt.close("all")
    return results
