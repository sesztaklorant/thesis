"""
aspects.py — Aspektus-alapú sentiment elemzés (ABSA), RQ3 és RQ4.

Két szint van, és ezt a dolgozatban is érdemes így felépíteni:

  1. LEXIKON-ALAPÚ ASPEKTUS-DETEKTÁLÁS (ebben a fájlban)
     Kulcsszólisták aspektusonként + mondatra bontás. Egyszerű, gyors,
     átlátható, és azonnal ad számokat az RQ3-ra ("mely aspektusok
     kötődnek leggyakrabban a negatív sentimenthez?").
     Ez a BASELINE, amihez a transformer modellt hasonlítod.

  2. TRANSFORMER-ALAPÚ ABSA (a 04-es notebookban, HRAST-on tanítva)
     Az aspektust és a mondatot párba állítva adjuk a BERT-nek:
         [CLS] room [SEP] the room was extremely hot [SEP] -> negative
     Ez a "sentence-pair" formátum a SemEval-2014 óta bevett módszer.

A kettő összehasonlítása önmagában is jó dolgozat-anyag: megmutatod, hogy
a kulcsszavas megközelítés hol bukik el (tagadás, irónia, implicit
aspektusok), és hogy a transformer ezt mennyivel oldja meg jobban.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .config import CFG

log = logging.getLogger(__name__)


# ==========================================================================
# 1) ASPEKTUS-LEXIKON
# ==========================================================================
# FONTOS: ezt a listát a DOLGOZATBAN közölni kell (függelékben), mert ez
# egy módszertani döntés. Bővítsd nyugodtan az adatod alapján — a 01-es
# notebook gyakorisági elemzése megmutatja, mely szavak hiányoznak.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "room": [
        "room", "rooms", "bedroom", "suite", "bed", "beds", "mattress",
        "pillow", "pillows", "sheets", "linen", "air conditioning", "aircon",
        "a/c", "heating", "radiator", "window", "balcony", "view",
    ],
    "staff": [
        "staff", "reception", "receptionist", "employee", "employees",
        "manager", "service", "housekeeping", "concierge", "waiter",
        "waitress", "front desk", "host", "team", "personnel",
    ],
    "cleanliness": [
        "clean", "cleanliness", "dirty", "dust", "dusty", "stain", "stains",
        "smell", "smelly", "odour", "odor", "mold", "mould", "hygiene",
        "hygienic", "filthy", "spotless", "tidy", "cockroach", "bed bugs",
    ],
    "facilities": [
        "facility", "facilities", "gym", "fitness", "spa", "sauna",
        "elevator", "lift", "restaurant", "bar", "lobby", "reception area",
        "amenities", "kettle", "fridge", "minibar", "tv", "television",
    ],
    "location": [
        "location", "located", "situated", "central", "centre", "center",
        "downtown", "station", "metro", "bus", "airport", "walking distance",
        "close to", "near", "nearby", "neighbourhood", "neighborhood",
        "transport", "beach", "city center", "city centre",
    ],
    "breakfast": [
        "breakfast", "buffet", "coffee", "croissant", "morning meal",
        "continental breakfast", "food", "meal", "dinner", "lunch",
    ],
    "pool": ["pool", "swimming", "jacuzzi", "hot tub"],
    "bathroom": [
        "bathroom", "shower", "toilet", "bath", "sink", "towel", "towels",
        "hot water", "water pressure", "toiletries", "hairdryer",
    ],
    "parking": ["parking", "car park", "garage", "valet"],
    "noise": [
        "noise", "noisy", "loud", "quiet", "soundproof", "soundproofing",
        "thin walls", "street noise", "traffic noise",
    ],
    "wifi": ["wifi", "wi-fi", "wireless", "internet", "connection", "signal"],
    "value_for_money": [
        "price", "prices", "expensive", "cheap", "value", "money",
        "overpriced", "worth", "cost", "affordable", "rate", "charge",
        "extra charge", "hidden fee",
    ],
}

# Előre lefordított regexek — szóhatárral, hogy a "bar" ne találjon a
# "barely" szóba. Többszavas kifejezéseknél a szóköz is illeszkedik.
_ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}


# ==========================================================================
# 2) MONDATRA BONTÁS
# ==========================================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """
    Egyszerű mondatra bontás. Hotel reviewknál ez általában elég jó, mert
    a szöveg rövid és egyszerű szerkezetű.

    Ha pontosabb kell (rövidítések, tizedespontok), használd az NLTK-t:
        import nltk; nltk.download("punkt")
        from nltk.tokenize import sent_tokenize
    """
    if not isinstance(text, str):
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if len(p.split()) >= min_words]


def explode_by_polarity(
    df: pd.DataFrame,
    pos_col: str = "review_positive",
    neg_col: str = "review_negative",
    min_words: int = 3,
) -> pd.DataFrame:
    """
    Mondat-szintű adat a POLARITÁS ISMERETÉBEN — az adathalmaz szerkezetéből.

    Miért sokkal jobb ez, mint a review-szintű címke rávetítése?
    ------------------------------------------------------------
    Az `explode_to_sentences()` minden mondatra a REVIEW összpontszámából
    képzett címkét teszi. Ez közelítés, és pont a dolgozatod kiinduló
    példájánál bukik el: egy 6,5 pontos ("neutral") review tartalmazhat egy
    egyértelműen pozitív és egy egyértelműen negatív mondatot is.

    A Booking-adat viszont KÉT KÜLÖN MEZŐBEN tárolja a dicséretet és a
    panaszt. Ha a `review_positive` mezőből származó mondatot pozitívnak,
    a `review_negative` mezőből származót negatívnak vesszük, akkor a
    mondat-szintű címke nem becslés, hanem magából az adatgyűjtés
    szerkezetéből következik: a vendég maga döntötte el, melyik mezőbe írja.

    Ez ~200 000 reviewból több százezer mondat-szintű, aspektus-annotálható
    példát ad — nagyságrendekkel többet, mint a HRAST 23 113 mondata.

    ⚠ KORLÁTOK, amiket a dolgozatban le KELL írni:
      1. Ez a mező eredete, nem emberi annotáció. Előfordul, hogy valaki a
         pozitív mezőbe ír enyhe kritikát ("great, although a bit small").
         Zajos címke, de szisztematikusan nem torzít.
      2. NINCS semleges osztály — a mezőstruktúra bináris. A 3-osztályos
         feladathoz ez nem elég, a review-szintű címkézés marad.
      3. A HRAST továbbra is kell: az VALÓDI, emberi aspektus-annotációt
         tartalmaz, amin a modellt KIÉRTÉKELNI lehet. Ez itt tanításra és
         nagy mintás elemzésre jó, kiértékelési etalonnak nem.

    Visszatér
    ---------
    Soronként egy mondat: sentence_id, review_id, sentence, polarity
    ("positive" / "negative"), source_field, + a metaadat-oszlopok.
    """
    missing = {pos_col, neg_col} - set(df.columns)
    if missing:
        raise KeyError(
            f"Hiányzó oszlop(ok): {sorted(missing)}. Ez a függvény a Booking "
            "külön pozitív/negatív mezőit igényli — ellenőrizd, hogy a loader "
            "átvitte-e őket."
        )

    meta = [c for c in ["review_id", "score", "sentiment", "dataset",
                        "accommodation_type", "accommodation_country",
                        "accommodation_star_rating", "guest_type"]
            if c in df.columns]

    frames = []
    for col, polarity in [(pos_col, "positive"), (neg_col, "negative")]:
        part = df[meta + [col]].copy()
        part["sentence"] = part[col].map(lambda t: split_sentences(t, min_words))
        part = part.drop(columns=[col]).explode("sentence").dropna(subset=["sentence"])
        part["polarity"] = polarity
        part["source_field"] = col
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)
    out["sentence"] = out["sentence"].astype(str).str.strip()
    out = out[out["sentence"].str.len() > 0].reset_index(drop=True)

    if "review_id" in out.columns:
        out["sentence_id"] = (out["review_id"].astype(str) + "_" +
                              out["polarity"].str[:3] + "_" +
                              out.groupby(["review_id", "polarity"]).cumcount().astype(str))

    log.info("Mondatok polaritás szerint: %s",
             out["polarity"].value_counts().to_dict())
    return out


def aspect_polarity_table(
    sent_df: pd.DataFrame,
    polarity_col: str = "polarity",
    min_count: int = 50,
) -> pd.DataFrame:
    """
    Aspektusonként: hányszor említik dicséretként, hányszor panaszként?

    Ez az RQ3 LEGTISZTÁBB válasza, mert a polaritás nem becsült, hanem az
    adat szerkezetéből jön (lásd `explode_by_polarity`).

    A `panasz_arány` oszlop: az adott aspektus említéseinek hány százaléka
    esik a negatív mezőbe. 0,5 felett panaszhajtó, alatta dicsérethajtó.
    """
    base = (sent_df[polarity_col].astype(str) == "negative").mean()
    rows = []
    for aspect in CFG.aspects:
        col = f"aspect_{aspect}"
        if col not in sent_df.columns:
            continue
        sub = sent_df[sent_df[col] == 1]
        if len(sub) < min_count:
            continue
        neg = (sub[polarity_col].astype(str) == "negative").mean()
        rows.append({
            "aspektus": aspect,
            "említések": len(sub),
            "dicséretként": 1 - neg,
            "panaszként": neg,
            "panasz_lift": neg / base if base else float("nan"),
        })

    cols = ["aspektus", "említések", "dicséretként", "panaszként", "panasz_lift"]
    if not rows:
        log.warning("Egyetlen aspektus sem érte el a min_count=%d küszöböt.", min_count)
        return pd.DataFrame(columns=cols).set_index("aspektus")

    return (pd.DataFrame(rows)[cols]
            .set_index("aspektus")
            .sort_values("panasz_lift", ascending=False)
            .round(4))


def explode_to_sentences(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Review-szintű táblából mondat-szintű táblát csinál.

    Ez az ABSA első lépése: az aspektus-sentiment párokat mondatszinten
    értelmezzük, mert egy review több, egymásnak ellentmondó véleményt
    tartalmazhat (pont ez a dolgozat kiinduló példája!).
    """
    out = df.copy()
    out["sentences"] = out[text_col].map(split_sentences)
    out = out.explode("sentences").dropna(subset=["sentences"])
    out = out.rename(columns={"sentences": "sentence"}).reset_index(drop=True)
    out["sentence_id"] = out["review_id"].astype(str) + "_s" + out.groupby("review_id").cumcount().astype(str)
    return out


# ==========================================================================
# 3) ASPEKTUS-DETEKTÁLÁS
# ==========================================================================
def detect_aspects(text: str) -> list[str]:
    """Visszaadja, mely aspektusok kulcsszavai fordulnak elő a szövegben."""
    if not isinstance(text, str) or not text:
        return []
    return [a for a, pat in _ASPECT_PATTERNS.items() if pat.search(text)]


def add_aspect_columns(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Aspektusonként egy 0/1 oszlopot ad hozzá (`aspect_room`, `aspect_staff`, ...),
    plusz egy listát az összes talált aspektussal.
    """
    out = df.copy()
    out["aspects"] = out[text_col].map(detect_aspects)
    for aspect in CFG.aspects:
        out[f"aspect_{aspect}"] = out["aspects"].map(lambda lst, a=aspect: int(a in lst))
    out["n_aspects"] = out["aspects"].map(len)
    return out


# ==========================================================================
# 4) ASPEKTUS × SENTIMENT ELEMZÉS  (RQ3 KÖZVETLEN VÁLASZA)
# ==========================================================================
def aspect_sentiment_table(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    min_count: int = 30,
) -> pd.DataFrame:
    """
    Aspektusonként megadja, hogy hány mondatban fordul elő, és azokban
    milyen a sentiment megoszlása.

    Ez a táblázat a dolgozat egyik LEGFONTOSABB eredménye: ebből derül ki,
    hogy pl. a 'noise' és a 'cleanliness' aránytalanul gyakran negatív,
    míg a 'location' túlnyomórészt pozitív.

    A "negatív többlet" (neg_lift) oszlop: mennyivel gyakoribb a negatív
    sentiment ennél az aspektusnál, mint az adathalmaz átlagában.
    1.0 felett = az aspektus panaszhajtó; 1.0 alatt = elégedettséghajtó.
    """
    base_neg = (df[sentiment_col].astype(str) == "negative").mean()
    base_pos = (df[sentiment_col].astype(str) == "positive").mean()

    rows = []
    for aspect in CFG.aspects:
        col = f"aspect_{aspect}"
        if col not in df.columns:
            continue
        sub = df[df[col] == 1]
        if len(sub) < min_count:
            continue
        dist = sub[sentiment_col].astype(str).value_counts(normalize=True)
        neg, neu, pos = (dist.get("negative", 0.0),
                         dist.get("neutral", 0.0),
                         dist.get("positive", 0.0))
        rows.append({
            "aspektus": aspect,
            "említések": len(sub),
            "említési_arány": len(sub) / len(df),
            "negative": neg,
            "neutral": neu,
            "positive": pos,
            "neg_lift": neg / base_neg if base_neg else float("nan"),
            "pos_lift": pos / base_pos if base_pos else float("nan"),
        })

    cols = ["aspektus", "említések", "említési_arány",
            "negative", "neutral", "positive", "neg_lift", "pos_lift"]

    if not rows:
        # Egyetlen aspektus sem érte el a min_count küszöböt. Ne álljon meg a
        # notebook — üres, de helyes szerkezetű táblát adunk vissza.
        log.warning(
            "Egyetlen aspektus sem érte el a min_count=%d küszöböt. "
            "Csökkentsd a min_count értéket, vagy növeld a mintát.", min_count
        )
        return pd.DataFrame(columns=cols).set_index("aspektus")

    return (pd.DataFrame(rows)[cols]
            .set_index("aspektus")
            .sort_values("neg_lift", ascending=False)
            .round(4))


def plot_aspect_sentiment(table: pd.DataFrame, save_as: str | None = None):
    """Halmozott oszlopdiagram: aspektusonkénti sentiment-megoszlás."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    if table.empty:
        log.warning("Üres az aspektus-tábla, nincs mit ábrázolni.")
        return None

    from .config import SENTIMENT_COLORS, INK

    data = table[["negative", "neutral", "positive"]].sort_values("negative", ascending=True)
    # A tengely KATEGORIKUS: az indexet stringgé kell alakítani, különben a
    # matplotlib számértékként helyezi el a sávokat (numerikus indexnél ez
    # egymásra csúszó oszlopokat okoz).
    labels = [str(i) for i in data.index]

    fig, ax = plt.subplots(figsize=(8.5, max(3.6, 0.42 * len(data) + 2.2)))
    left = pd.Series(0.0, index=data.index)
    for col in ["negative", "neutral", "positive"]:
        ax.barh(labels, data[col], left=left, label=col,
                color=SENTIMENT_COLORS[col], height=0.72,
                edgecolor=INK["surface"], linewidth=1.5)
        left += data[col]

    ax.set_xlabel("A sentiment megoszlása az adott aspektust említő mondatokban")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Aspektusonkénti sentiment-megoszlás", loc="left", pad=34,
                 color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 5) TEXTUÁLIS BIZONYÍTÉK (RQ4 baseline)
# ==========================================================================
_NEGATORS = {"no", "not", "never", "without", "lack", "lacked", "lacking",
             "n't", "nothing", "none", "hardly", "barely"}


def extract_evidence(sentence: str, aspect: str, window: int = 4) -> list[str]:
    """
    Kiemeli az aspektus-kulcsszó körüli szókörnyezetet — ez a legegyszerűbb
    "textuális bizonyíték" (a dolgozat 10. fejezetének 3. szintje).

    Példa:
        extract_evidence("the room was extremely hot because there was no air conditioning", "room")
        -> ["the room was extremely hot because"]

    A transformer-alapú változat (04-es notebook) ennél sokkal jobb lesz,
    mert az attention-súlyokat vagy egy token-osztályozó fejet használ.
    Ez a függvény a BASELINE, amihez viszonyítasz.
    """
    pat = _ASPECT_PATTERNS.get(aspect)
    if pat is None or not isinstance(sentence, str):
        return []

    tokens = sentence.split()
    lowered = [t.lower().strip(".,!?;:") for t in tokens]
    kws = {k.lower() for k in ASPECT_KEYWORDS[aspect] if " " not in k}

    spans = []
    for i, tok in enumerate(lowered):
        if tok in kws:
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            spans.append(" ".join(tokens[lo:hi]))
    return spans


def has_negation(text: str) -> bool:
    """Van-e tagadás a szövegrészletben? A kulcsszavas ABSA fő gyengesége."""
    if not isinstance(text, str):
        return False
    toks = {t.lower().strip(".,!?;:") for t in text.split()}
    return bool(toks & _NEGATORS) or "n't" in text.lower()


# ==========================================================================
# 6) SENTENCE-PAIR FORMÁTUM A TRANSFORMER ABSA-HOZ
# ==========================================================================
def build_absa_pairs(
    df: pd.DataFrame,
    sentence_col: str = "sentence",
    aspect_label_col: str | None = None,
) -> pd.DataFrame:
    """
    Előkészíti az adatot a transformer ABSA-hoz (04-es notebook).

    Kimenet soronként: (aspektus, mondat, [címke])
    A tokenizálás a notebookban így néz majd ki:

        tokenizer(batch["aspect"], batch["sentence"], truncation=True)

    A HuggingFace tokenizer két argumentummal automatikusan a
    [CLS] aspect [SEP] sentence [SEP] formátumot állítja elő.
    """
    rows = []
    for _, r in df.iterrows():
        for aspect in r.get("aspects", []) or []:
            row = {
                "sentence_id": r.get("sentence_id"),
                "aspect": aspect.replace("_", " "),
                "sentence": r[sentence_col],
                "evidence": " | ".join(extract_evidence(r[sentence_col], aspect)),
            }
            if aspect_label_col and aspect_label_col in r:
                row["label"] = r[aspect_label_col]
            rows.append(row)
    return pd.DataFrame(rows)
