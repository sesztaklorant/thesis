"""
descriptives.py — Leíró statisztikák és kereszttáblák a megtisztított adaton.

A dolgozat 9. fejezete (Exploratory Data Analysis).

Mire való ez a modul?
---------------------
Az EDA után jön a kérdés: "mit tudok mondani az adatról azon túl, hogy
hány sor van benne?" A Booking.com adathalmaz metaadatai (ország,
szállástípus, vendégtípus, csillagbesorolás) lehetővé teszik, hogy a
sentimentet ezek MENTÉN vizsgáld. Ez az, amit egy szállodalánc elemzője
is megcsinálna — és pont ezért érdekes a dolgozatban.

A központi mutató: LIFT
-----------------------
Nem elég azt mondani, hogy "a hostelek 35%-a negatív". Tudni kell, hogy
ez sok vagy kevés. A lift ezt adja meg:

    lift = az adott csoport negatív aránya / a teljes adathalmaz negatív aránya

    lift = 1,0  ->  pontosan átlagos
    lift = 1,4  ->  40%-kal gyakrabban negatív az átlagnál
    lift = 0,7  ->  30%-kal ritkábban

Ez a mutató összehasonlíthatóvá tesz különböző méretű csoportokat, és
egyetlen számban mondja meg, hol van a probléma.

Ábrák
-----
Minden ábra a CFG színpalettáját használja (piros-kék divergáló, nem
piros-zöld — lásd a config.py indoklását a színtévesztésről).
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
    Kategórianév ábrázoláshoz. A lebegőpontos egész értékeket (3.0, 4.0)
    egésszé alakítja, mert a "3.0 csillag" felirat zavaró.
    """
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v)


def prepare_star_rating(df: pd.DataFrame,
                        col: str = "accommodation_star_rating") -> pd.DataFrame:
    """
    A csillagbesorolás 0 értéke NEM "nulla csillag", hanem "nincs besorolás".

    Ha ezt nem kezeljük, a 0 egy önálló kategóriaként jelenik meg a
    kereszttáblákban, és a rendezésnél a legrosszabb besorolásnak látszik —
    miközben valójában hiányzó adat. A te mintádban ez több tízezer sor.

    Ez a függvény külön "n. a." kategóriát csinál belőle, így a
    besorolás szerinti elemzés csak a ténylegesen besorolt szállásokra
    vonatkozik, a többi pedig láthatóan elkülönül.
    """
    if col not in df.columns:
        return df
    out = df.copy()
    n_zero = int((pd.to_numeric(out[col], errors="coerce") == 0).sum())
    out[col] = pd.to_numeric(out[col], errors="coerce").map(
        lambda v: "n. a." if (pd.isna(v) or v == 0) else _fmt_cat(v)
    )
    if n_zero:
        log.info("Csillagbesorolás: %d sor (%.1f%%) 0 értékű volt -> 'n. a.' "
                 "kategóriába került.", n_zero, 100 * n_zero / len(out))
    return out


# ==========================================================================
# 1) ÁTTEKINTŐ TÁBLÁZAT
# ==========================================================================
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Egysoros összefoglaló az adathalmazról — ez megy a dolgozat 4. fejezetébe
    ("Data Sources") és a 9. fejezet elejére.
    """
    rows: list[tuple[str, object]] = [
        ("Reviewk száma", f"{len(df):,}"),
    ]

    if "accommodation_id" in df.columns:
        rows.append(("Egyedi szálláshelyek", f"{df['accommodation_id'].nunique():,}"))
    if "accommodation_country" in df.columns:
        rows.append(("Országok száma", f"{df['accommodation_country'].nunique():,}"))
    if "accommodation_type" in df.columns:
        rows.append(("Szállástípusok", f"{df['accommodation_type'].nunique():,}"))

    rows += [
        ("Átlagos pontszám", f"{df['score'].mean():.2f}"),
        ("Medián pontszám", f"{df['score'].median():.2f}"),
        ("Pontszám szórása", f"{df['score'].std():.2f}"),
    ]

    if "n_words" in df.columns:
        rows += [
            ("Átlagos reviewhossz (szó)", f"{df['n_words'].mean():.1f}"),
            ("Medián reviewhossz (szó)", f"{df['n_words'].median():.0f}"),
        ]

    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts(normalize=True)
        for label in CFG.labels:
            rows.append((f"{label} aránya", f"{vc.get(label, 0):.1%}"))

    return pd.DataFrame(rows, columns=["Mutató", "Érték"]).set_index("Mutató")


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Hiányzó értékek oszloponként. Üres táblát ad vissza, ha nincs hiány."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame(columns=["hiányzó", "arány"])
    return pd.DataFrame({"hiányzó": miss, "arány": (miss / len(df)).round(4)})


def numeric_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Numerikus oszlopok leíró statisztikája, percentilisekkel."""
    cols = cols or [c for c in ["score", "n_words", "n_chars", "room_nights",
                                "accommodation_score", "accommodation_star_rating",
                                "review_helpful_votes"]
                    if c in df.columns]
    return (df[cols]
            .describe(percentiles=[.05, .25, .5, .75, .95])
            .T.round(2))


# ==========================================================================
# 2) KERESZTTÁBLA: SENTIMENT EGY DIMENZIÓ MENTÉN
# ==========================================================================
def sentiment_by(
    df: pd.DataFrame,
    dim: str,
    min_count: int = 200,
    top_n: int | None = 15,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Sentiment-megoszlás egy kategorikus dimenzió mentén, lift értékekkel.

    Paraméterek
    -----------
    dim       : az oszlop neve (pl. "accommodation_country", "guest_type")
    min_count : ennél kevesebb reviewval rendelkező kategóriát eldobunk.
                MIÉRT? Mert 12 reviewból számolt 50%-os negatív arány
                statisztikai zaj, nem megállapítás. A dolgozatban ezt a
                küszöböt meg kell említeni.
    top_n     : csak a legtöbb reviewval rendelkező N kategóriát tartjuk meg.

    Visszatér
    ---------
    Index: a kategóriák. Oszlopok:
        n           — reviewk száma
        részarány   — a teljes adathalmaz hány százaléka
        negative / neutral / positive — a sentiment megoszlása a csoporton belül
        neg_lift    — hányszorosa az átlagos negatív aránynak
        átlagpont   — átlagos vendégpontszám
    """
    if dim not in df.columns:
        raise KeyError(f"Nincs ilyen oszlop: {dim}. Elérhető: {list(df.columns)}")

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
    tab["átlagpont"] = g["score"].mean()
    tab["részarány"] = tab["n"] / len(df)

    tab = tab[tab["n"] >= min_count]
    if top_n:
        tab = tab.nlargest(top_n, "n")

    cols = ["n", "részarány", *CFG.labels, "neg_lift", "pos_lift", "átlagpont"]
    return tab[cols].sort_values("neg_lift", ascending=False).round(4)


def plot_sentiment_by(
    table: pd.DataFrame,
    title: str,
    xlabel: str = "A sentiment megoszlása a csoporton belül",
    save_as: str | None = None,
    show_n: bool = True,
    figsize: tuple[float, float] = (9, None),
):
    """
    100%-ra halmozott vízszintes oszlopdiagram.

    Miért EZ a forma?
      - A kérdés "rész-egész arány több kategórián keresztül" -> halmozott sáv.
      - Vízszintes, mert a kategórianevek (országnevek, szállástípusok)
        hosszúak; függőlegesen egymásra csúsznának.
      - 100%-ra normalizálva, mert a csoportok mérete nagyon eltérő — a
        nyers darabszám elnyomná a kicsiket. A darabszám a sáv végén,
        szövegként jelenik meg.
      - A szegmensek közt 2px-es felületszínű rés: így a határvonal akkor is
        látszik, ha két szomszédos szín hasonló tónusú (pl. nyomtatásban).
    """
    import matplotlib.pyplot as plt

    data = table[list(CFG.labels)].sort_values("negative", ascending=True)

    # A tengely KATEGORIKUS. Ha az index numerikus (pl. csillagbesorolás:
    # 0.0, 2.0, 3.0, 3.5 ...), a matplotlib SZÁMÉRTÉKKÉNT helyezi el a
    # sávokat a tengelyen — ilyenkor egymásra csúsznak, és a feliratok
    # elcsúsznak. Stringgé alakítva minden kategória egyenlő sávot kap.
    ypos = [_fmt_cat(i) for i in data.index]

    # A magasság a kategóriák számából jön, de van alsó korlát: a cím és a
    # fölötte ülő jelmagyarázat helyet kér, különben a tight_layout panaszkodik.
    h = figsize[1] or max(3.4, 0.42 * len(data) + 2.2)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    left = pd.Series(0.0, index=data.index)
    for label in CFG.labels:
        ax.barh(
            ypos, data[label], left=left.values, label=label,
            color=SENTIMENT_COLORS[label], height=0.72,
            edgecolor=INK["surface"], linewidth=1.5,   # a 2px-es rés
        )
        left += data[label]

    # Közvetlen felirat csak a negatív szegmensen — ez az, amit olvasni
    # akarunk. Minden szegmensre írni számot zsúfolt és olvashatatlan lenne.
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

    # A jelmagyarázat a cím ALÁ kerül, nem az ábra alá. Így kevés kategóriánál
    # sem szorul ki a helyéről (a tight_layout különben figyelmeztetne).
    ax.set_title(title, loc="left", pad=34, color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) HŐTÉRKÉP: KÉT DIMENZIÓ EGYÜTT
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
    Két dimenzió kereszttáblája: mekkora a negatív arány minden cellában?

    Visszatér: (lift-tábla, darabszám-tábla, átlagos negatív arány)

    A lift-tábla azt mutatja, hányszorosa az adott cella negatív aránya a
    teljes adathalmaz átlagának. Így a nagyon eltérő méretű cellák is
    összehasonlíthatók.
    """
    sent = df[sentiment_col].astype(str)
    base_neg = (sent == "negative").mean()

    work = df.assign(_neg=(sent == "negative").astype(float))

    keep_r = work[rows].value_counts().nlargest(top_rows).index
    keep_c = work[cols].value_counts().nlargest(top_cols).index
    work = work[work[rows].isin(keep_r) & work[cols].isin(keep_c)]

    counts = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="size")
    neg_share = work.pivot_table(index=rows, columns=cols, values="_neg", aggfunc="mean")

    # A megbízhatatlanul kicsi cellákat kimaszkoljuk
    neg_share = neg_share.where(counts >= min_count)
    lift = neg_share / base_neg if base_neg else neg_share

    # A teljesen üres sorok/oszlopok csak helyet foglalnának az ábrán
    lift = lift.dropna(how="all", axis=0).dropna(how="all", axis=1)
    counts = counts.reindex(index=lift.index, columns=lift.columns)

    if lift.empty:
        raise ValueError(
            f"Minden cella kimaszkolva (min_count={min_count}). "
            "Csökkentsd a min_count értéket, vagy válassz kevesebb kategóriát "
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
    Divergáló hőtérkép: kék = az átlagnál KEVESEBB negatív, piros = TÖBB.

    Miért divergáló és nem szekvenciális?
      Mert a kérdés nem "mennyi?", hanem "az átlaghoz képest jobb vagy
      rosszabb?" — ez polaritás, aminek van értelmes közepe (lift = 1,0).
      Egy szekvenciális skálán nem látszana, hol van az átlag.

    A szürke cellák: túl kevés review, nem értelmezhető. Ezt a dolgozatban
    is jelölni kell, nem szabad üresen hagyni magyarázat nélkül.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div_br", DIVERGING_BLUE_RED)
    cmap.set_bad(INK["grid"])

    vals = lift.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("Minden cella kimaszkolva — csökkentsd a min_count értéket.")

    # A skála a lift = 1,0 (átlag) körül szimmetrikus, DE alulról 0-nál
    # levágva: egy arányszám nem lehet negatív, és egy 0 alá nyúló
    # színskála értelmezhetetlen lenne az olvasónak.
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
            # Sötét cellán fehér, világoson fekete szöveg
            dark = abs(v - 1) > span * 0.55
            txt = f"{v:.2f}"
            if annotate_n:
                txt += f"\nn={int(counts.iloc[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK["primary"])

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(f"Negatív arány / átlag ({base_neg:.1%})", fontsize=9.5)
    cb.outline.set_visible(False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 4) EGYSZERŰ GYAKORISÁGI ÁBRA
# ==========================================================================
def plot_top_categories(
    df: pd.DataFrame,
    col: str,
    top_n: int = 12,
    title: str | None = None,
    save_as: str | None = None,
):
    """
    Egyszerű vízszintes oszlopdiagram: melyik kategóriából van a legtöbb?

    Egy adatsor -> NINCS jelmagyarázat (a cím megnevezi), és a számok
    közvetlenül a sávok végén állnak.
    """
    import matplotlib.pyplot as plt

    vc = df[col].value_counts().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.38 * len(vc) + 1.2)))

    ax.barh(vc.index.astype(str), vc.values, color="#2a78d6", height=0.72)
    for y, v in enumerate(vc.values):
        ax.text(v * 1.01, y, f"{v:,}", va="center", fontsize=9,
                color=INK["secondary"])

    ax.set_xlabel("Reviewk száma")
    ax.set_title(title or f"Leggyakoribb értékek — {col}", loc="left",
                 pad=12, color=INK["primary"])
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 5) ÖSSZEFOGLALÓ FUTTATÁS
# ==========================================================================
def run_all_crosstabs(
    df: pd.DataFrame,
    dims: dict[str, str] | None = None,
    min_count: int = 200,
    top_n: int = 12,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Végigfuttatja a kereszttáblákat minden elérhető dimenzión, elmenti a
    táblázatokat és az ábrákat.

    dims: {oszlopnév: "magyar megnevezés az ábra címéhez"}
          None esetén az alapértelmezett készlet, amennyi az adatban van.

    Visszatér: {oszlopnév: tábla}
    """
    import matplotlib.pyplot as plt

    default_dims = {
        "accommodation_country": "szálláshely országa",
        "accommodation_type": "szállástípus",
        "guest_type": "vendégtípus",
        "accommodation_star_rating": "csillagbesorolás",
        "guest_country": "vendég országa",
    }
    dims = dims or {k: v for k, v in default_dims.items() if k in df.columns}

    results: dict[str, pd.DataFrame] = {}
    for col, label in dims.items():
        if col not in df.columns:
            log.warning("Kihagyva (nincs ilyen oszlop): %s", col)
            continue
        try:
            tab = sentiment_by(df, col, min_count=min_count, top_n=top_n)
        except Exception as e:                       # noqa: BLE001
            log.warning("Kihagyva (%s): %s", col, e)
            continue
        if tab.empty:
            log.warning("Kihagyva (üres a min_count=%d szűrés után): %s", min_count, col)
            continue

        results[col] = tab
        if save:
            save_table(tab, f"01_crosstab_{col}")
            plot_sentiment_by(tab, f"Sentiment-megoszlás — {label}",
                              save_as=f"01_crosstab_{col}")
            plt.close("all")
        log.info("%-28s %d kategória, neg_lift tartomány %.2f–%.2f",
                 col, len(tab), tab["neg_lift"].min(), tab["neg_lift"].max())

    return results
