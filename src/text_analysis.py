"""
text_analysis.py — Megkülönböztető kifejezések és tagadáselemzés.

Erre a kérdésre válaszol: "mikor rakjuk össze a szavakat, hogy jelentést
kapjanak — 'good' vs 'not good'?"

A probléma a puszta gyakorisággal
----------------------------------
A 01-es notebook első változata osztályonként a LEGGYAKORIBB szavakat
mutatta. Az eredmény szinte használhatatlan volt: mindhárom osztályban
ugyanazok a szavak vezettek (room, location, staff, good). Ez nem hiba,
hanem a módszer korlátja — a gyakori szavak mindenhol gyakoriak.

Amit tudni akarunk, az más kérdés:

    NEM: "mely szavak gyakoriak a negatív reviewkban?"
    HANEM: "mely szavak gyakoribbak a negatívban, MINT A TÖBBIBEN?"

Ez a megkülönböztető (distinctive) kifejezések kérdése, és külön mérőszám
kell hozzá.

A használt módszer: súlyozott log-odds arány
---------------------------------------------
Monroe, Colaresi és Quinn (2008) "Fightin' Words" módszere, ami a
politikai szövegelemzés egyik bevett eszköze, és sentimentre is jól
működik. Két lépésből áll:

1. LOG-ODDS: mennyivel valószínűbb a szó az egyik osztályban, mint a
   másikban? A logaritmus szimmetrikussá teszi (a kétszer gyakoribb és a
   feleannyira gyakori azonos távolságra kerül a nullától).

2. Z-ÉRTÉKKÉ NORMÁLÁS: a ritka szavak log-odds értéke bizonytalan — egy
   3-szor előforduló szó aránya véletlenül is kiugorhat. A becslés
   szórásával osztva a ritka szavak visszaszorulnak, és a lista tetején
   olyan kifejezések maradnak, amelyek gyakoriak ÉS megkülönböztetőek.

Hivatkozás a dolgozatba:
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
# 1) MEGKÜLÖNBÖZTETŐ KIFEJEZÉSEK
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
    Osztályonként megadja a legjellemzőbb kifejezéseket.

    Paraméterek
    -----------
    ngram_range : (1,1) csak szavak, (1,2) szavak + szópárok, (1,3) hármasok is.
        A SZÓPÁROK a lényegesek: a "not clean" csak bigramként létezik.
    min_count   : ennél ritkább kifejezést eldobunk (zaj).
    alpha       : simítási paraméter. Megakadályozza a nullával osztást, és
        tompítja a nagyon ritka kifejezéseket.

    Visszatér
    ---------
    Hosszú formátumú tábla: sentiment | rang | kifejezés | z_ertek |
    gyakorisag_osztalyban | gyakorisag_maskor | arany
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

        y = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)   # osztályon belül
        rest = total - y                                            # a többi osztály

        n_y, n_rest = y.sum(), rest.sum()

        # Simított arányok
        p_y = (y + alpha) / (n_y + alpha * V)
        p_r = (rest + alpha) / (n_rest + alpha * V)

        # Log-odds arány
        lo = np.log(p_y / (1 - p_y)) - np.log(p_r / (1 - p_r))

        # A becslés szórása, és a belőle képzett z-érték
        var = 1.0 / (y + alpha) + 1.0 / (rest + alpha)
        z = lo / np.sqrt(var)

        # Csak a ténylegesen előforduló kifejezések érdekelnek
        ok = y >= max(min_count // 3, 3)
        idx = np.argsort(np.where(ok, z, -np.inf))[-top_n:][::-1]

        for rank, i in enumerate(idx, start=1):
            if not np.isfinite(z[i]):
                continue
            rows.append({
                "sentiment": cls,
                "rang": rank,
                "kifejezés": vocab[i],
                "z_érték": float(z[i]),
                "db_osztályban": int(y[i]),
                "db_máshol": int(rest[i]),
                "arány": float((y[i] / max(n_y, 1)) / max(rest[i] / max(n_rest, 1), 1e-12)),
            })

    log.info("Szókincs: %d kifejezés (%s-gram), összesen %d token",
             V, "-".join(map(str, ngram_range)), int(grand_total))
    return pd.DataFrame(rows)


def plot_distinctive_terms(
    table: pd.DataFrame,
    top_n: int = 15,
    save_as: str | None = None,
):
    """
    Osztályonként vízszintes oszlopdiagram a z-értékekkel.

    Egy adatsor panelenként -> nincs jelmagyarázat, a panel címe nevezi meg.
    """
    import matplotlib.pyplot as plt

    classes = [c for c in CFG.labels if c in set(table["sentiment"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 5.6))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = (table[table["sentiment"] == cls]
               .nsmallest(top_n, "rang")
               .sort_values("z_érték"))
        ax.barh(sub["kifejezés"], sub["z_érték"],
                color=SENTIMENT_COLORS[cls], height=0.72)
        ax.set_title(cls, loc="left", pad=10, color=INK["primary"])
        ax.set_xlabel("z-érték (megkülönböztető erő)")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.margins(x=0.10)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return axes


# ==========================================================================
# 2) TAGADÁSELEMZÉS
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
    Kigyűjti a TAGADÁSSAL kezdődő szópárokat, és megmutatja az
    osztályonkénti megoszlásukat.

    Ez a kérdésed közvetlen válasza: hány "not good", "not clean",
    "no hot water" fordul elő, és melyik osztályban?

    Visszatér: bigram | összes | negative/neutral/positive arány | neg_lift
    """
    from sklearn.feature_extraction.text import CountVectorizer

    neg_pattern = re.compile(r"^(" + "|".join(re.escape(n) for n in NEGATORS) + r") ")

    cv = CountVectorizer(ngram_range=(2, 2), min_df=min_count)
    X = cv.fit_transform(texts.fillna("").astype(str))
    vocab = np.array(cv.get_feature_names_out())

    keep = np.array([bool(neg_pattern.match(v)) for v in vocab])
    if not keep.any():
        log.warning("Nem találtam tagadással kezdődő szópárt. Lehet, hogy a "
                    "szövegoszlopból már kiestek a tagadószavak — ellenőrizd, "
                    "hogy keep_negations=True volt-e a tisztításnál.")
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

    df = pd.DataFrame({"bigram": vocab, "összes": total.astype(int)})
    for cls in CFG.labels:
        df[cls] = shares[cls] / np.maximum(total, 1)
    df["neg_lift"] = df["negative"] / base_neg if base_neg else np.nan

    return (df.nlargest(top_n, "összes")
              .set_index("bigram")
              .sort_values("neg_lift", ascending=False)
              .round(4))


def compare_with_without_negation(
    texts: pd.Series,
    labels: pd.Series,
    words: list[str] | None = None,
) -> pd.DataFrame:
    """
    A kérdésed magja: mennyit számít a "not" a szó előtt?

    Szavanként összeveti:
        "good"      hány százaléka esik negatív reviewba?
        "not good"  hány százaléka esik negatív reviewba?

    Ha a kettő között nagy a különbség, az SZÁMSZERŰ BIZONYÍTÉK arra, hogy
    az unigram-alapú (zsák-a-szavak) reprezentáció elveszít valamit —
    pontosan ez az RQ2 lényege.

    A táblázat mehet a dolgozat 6.2 fejezetébe.
    """
    words = words or ["good", "clean", "comfortable", "friendly", "helpful",
                      "nice", "quiet", "great", "big", "working", "hot water",
                      "problem", "recommend"]

    txt = texts.fillna("").astype(str).str.lower()
    lab = labels.astype(str)
    base = (lab == "negative").mean()

    # Nem-rögzítő csoport (?:...) kell, különben a pandas figyelmeztet, hogy
    # a mintában rögzítő csoport van, és az str.extract-et ajánlja helyette.
    neg_prefixes = "(?:" + "|".join(re.escape(n) for n in NEGATORS) + ")"

    rows = []
    for w in words:
        wq = re.escape(w)
        # Előfordul-e egyáltalán a szó? (szóhatárral, hogy a "good" ne
        # találjon a "goodbye"-ba)
        present = txt.str.contains(rf"(?<!\w){wq}(?!\w)", regex=True, na=False)
        # Előfordul-e TAGADÁS UTÁN közvetlenül?
        negated = txt.str.contains(rf"(?<!\w){neg_prefixes}\s+{wq}(?!\w)",
                                   regex=True, na=False)
        # "Önmagában" = szerepel, de nem tagadva. (Ha egy reviewban mindkét
        # alak előfordul, azt a tagadott csoportba soroljuk — óvatosabb,
        # mert a tagadás erősebb jel.)
        plain_only = present & ~negated

        if plain_only.sum() < 20 or negated.sum() < 20:
            continue

        p_neg_plain = (lab[plain_only] == "negative").mean()
        p_neg_negated = (lab[negated] == "negative").mean()

        rows.append({
            "szó": w,
            "db (önmagában)": int(plain_only.sum()),
            "negatív arány (önmagában)": p_neg_plain,
            "db (tagadva)": int(negated.sum()),
            "negatív arány (tagadva)": p_neg_negated,
            "különbség": p_neg_negated - p_neg_plain,
            "lift (tagadva)": p_neg_negated / base if base else np.nan,
        })

    if not rows:
        log.warning("Egyik szónál sem volt elég előfordulás az összevetéshez.")
        return pd.DataFrame()

    return (pd.DataFrame(rows)
            .set_index("szó")
            .sort_values("különbség", ascending=False)
            .round(4))


def plot_negation_effect(table: pd.DataFrame, save_as: str | None = None):
    """
    Párosított pontdiagram (dumbbell): minden szónál a puszta és a tagadott
    alak negatív aránya, vonallal összekötve.

    Miért ez a forma? Mert a kérdés PÁROS összehasonlítás — a vonal hossza
    közvetlenül mutatja a hatás nagyságát, és egyben látszik a kiindulási
    szint is. Két külön oszlopdiagram ugyanezt nehezebben olvashatóan adná.
    """
    import matplotlib.pyplot as plt

    if table.empty:
        log.warning("Üres tábla, nincs mit ábrázolni.")
        return None

    d = table.sort_values("különbség")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.42 * len(d) + 2.0)))

    ax.hlines(y, d["negatív arány (önmagában)"], d["negatív arány (tagadva)"],
              color=INK["axis"], lw=2, zorder=1)
    ax.scatter(d["negatív arány (önmagában)"], y, s=90, zorder=2,
               color=SENTIMENT_COLORS["positive"], label="a szó önmagában",
               edgecolor=INK["surface"], linewidth=1.5)
    ax.scatter(d["negatív arány (tagadva)"], y, s=90, zorder=2,
               color=SENTIMENT_COLORS["negative"], label="tagadással",
               edgecolor=INK["surface"], linewidth=1.5)

    ax.set_yticks(y, d.index)
    ax.set_xlabel("Az előfordulások mekkora hányada esik negatív reviewba")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("A tagadás hatása — ugyanaz a szó, más jelentés",
                 loc="left", pad=34, color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False)

    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) N-GRAM SZINTEK ÖSSZEHASONLÍTÁSA
# ==========================================================================
def ngram_level_comparison(
    texts: pd.Series,
    labels: pd.Series,
    min_count: int = 30,
) -> pd.DataFrame:
    """
    Mennyivel lesz "megkülönböztetőbb" a szókincs, ha szópárokat és
    hármasokat is megengedünk?

    A táblázat soronként egy n-gram szintet mutat: hány kifejezés van, és
    mekkora a top-50 kifejezés átlagos megkülönböztető ereje.

    Ez előrevetíti a 02-es notebook ablációs vizsgálatát: ott ugyanezt a
    kérdést MODELLTELJESÍTMÉNYBEN méred, itt a szókincs szintjén nézed meg.
    """
    rows = []
    for name, rng in [("unigram (1,1)", (1, 1)),
                      ("uni+bigram (1,2)", (1, 2)),
                      ("uni+bi+trigram (1,3)", (1, 3))]:
        t = distinctive_terms(texts, labels, ngram_range=rng,
                              top_n=50, min_count=min_count)
        if t.empty:
            continue
        n_multi = t["kifejezés"].str.contains(" ").sum()
        rows.append({
            "n-gram szint": name,
            "top-50 átlagos z": t.groupby("sentiment")["z_érték"].mean().mean(),
            "többszavas a top-50-ben": n_multi / len(t),
            "példa (negatív)": ", ".join(
                t[(t.sentiment == "negative") & (t["kifejezés"].str.contains(" "))]
                ["kifejezés"].head(3)
            ) or "—",
        })

    return pd.DataFrame(rows).set_index("n-gram szint").round(3)
