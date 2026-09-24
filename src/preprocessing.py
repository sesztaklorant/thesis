"""
preprocessing.py — Szövegtisztítás, címkézés, adatfelosztás.

A dolgozat 6.1. fejezete ("Data Preprocessing") ebből a modulból íródik.

Kulcsfontosságú elvi döntés, amit a dolgozatban is le kell írni:
-----------------------------------------------------------------
KÉT KÜLÖNBÖZŐ előfeldolgozási útvonal van, és ez SZÁNDÉKOS:

  (A) Klasszikus ML (TF-IDF + Naive Bayes / LogReg / SVM):
      agresszív normalizálás — kisbetűsítés, írásjelek eltávolítása,
      stopword-szűrés, lemmatizálás. A TF-IDF "zsák-a-szavak" modell,
      nem érti a szórendet, ezért a zaj csökkentése segít.

  (B) Transformer (BERT):
      MINIMÁLIS tisztítás. A BERT saját tokenizálót használ, és a
      kontextusból tanul. Ha eltávolítod a stopwordöket és a tagadást
      ("not clean" -> "clean"), akkor elrontod a jelentést és ROSSZABB
      eredményt kapsz. Ez egy gyakori kezdő hiba — és pont ezért jó
      anyag az RQ2 megválaszolásához.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from .config import CFG

log = logging.getLogger(__name__)


# ==========================================================================
# 1) SZÖVEGTISZTÍTÁS
# ==========================================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_text_minimal(text: str) -> str:
    """
    (B) útvonal — BERT-hez. Csak a valódi zajt távolítjuk el.
    Az írásjelek, a nagybetűk, az emodzsik és a tagadószavak MARADNAK.

    HTML-entitások feloldása
    ------------------------
    A Booking.com exportban a szöveg HTML-kódolt aposztrófokat tartalmaz:

        "wouldn&#39;t suggest walking"   ->   "wouldn't suggest walking"

    Ez NEM kozmetikai kérdés. Ha nem oldjuk fel:
      • a TF-IDF útvonalon a `[^a-z\\s']` szűrő kidobja a `&#39;` karaktereit,
        és a "wouldn&#39;t" -> "wouldn t" lesz — vagyis ELVESZIK A TAGADÁS,
        ami sentiment-elemzésnél a legrosszabb, ami történhet;
      • a BERT tokenizálója külön tokenekre bontja az entitást, és
        értelmetlen darabokat tanul.

    A `html.unescape` az összes elterjedt entitást kezeli (`&#39;`, `&amp;`,
    `&quot;`, `&lt;`, `&nbsp;` stb.), nem csak az aposztrófot.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)              # &#39; -> '   (EZ AZ ELSŐ LÉPÉS)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)          # valódi HTML tagek: <br>, <p>
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def clean_text_classic(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    keep_negations: bool = True,
) -> str:
    """
    (A) útvonal — TF-IDF-hez. Minden lépés külön kapcsolható,
    hogy az RQ2-höz ABLÁCIÓS vizsgálatot tudj futtatni:
    "mennyit számít, ha kikapcsolom a lemmatizálást?"

    keep_negations: ha True, a tagadószavakat (not, no, never, ...) NEM
        dobjuk ki a stopword-szűrésnél. Sentiment feladatnál ez általában
        segít — és önmagában is egy jó kis kísérlet a dolgozatba.
    """
    if not isinstance(text, str):
        return ""

    text = clean_text_minimal(text)
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = _NON_ALPHA_RE.sub(" ", text)

    tokens = text.split()

    if remove_stopwords or lemmatize:
        sw = _get_stopwords(keep_negations) if remove_stopwords else set()
        lem = _get_lemmatizer() if lemmatize else None
        out = []
        for t in tokens:
            if t in sw:
                continue
            out.append(lem.lemmatize(t) if lem else t)
        tokens = out

    return " ".join(tokens)


# --- NLTK erőforrások lusta betöltése (csak akkor, ha tényleg kell) ---
_STOPWORDS_CACHE: dict[bool, set[str]] = {}
_LEMMATIZER = None

NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "neither",
    "cannot", "cant", "wont", "dont", "didnt", "doesnt", "isnt",
    "wasnt", "werent", "hadnt", "hasnt", "havent", "wouldnt",
    "shouldnt", "couldnt", "aint",
}


def _get_stopwords(keep_negations: bool = True) -> set[str]:
    if keep_negations in _STOPWORDS_CACHE:
        return _STOPWORDS_CACHE[keep_negations]
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
    except Exception:
        log.warning("NLTK stopword lista nem elérhető, beépített minimális listát használok.")
        sw = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
              "it", "we", "i", "for", "on", "with", "at", "this", "that"}
    if keep_negations:
        sw = sw - NEGATION_WORDS
    _STOPWORDS_CACHE[keep_negations] = sw
    return sw


class _NaiveLemmatizer:
    """
    Tartalék megoldás, ha az NLTK/WordNet nem elérhető (pl. offline gépen).
    Csak a leggyakoribb angol többes számú és igei végződéseket kezeli.

    A dolgozatban FONTOS megemlíteni, ha ezt használtad az NLTK helyett —
    de a normál esetben az NLTK WordNetLemmatizer fut.
    """

    def lemmatize(self, w: str) -> str:
        for suf, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                          ("shes", "sh"), ("xes", "x")):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return w[: -len(suf)] + repl
        if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
            return w[:-1]
        return w


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                WordNetLemmatizer().lemmatize("tests")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("tests")  # ellenőrzés: tényleg működik-e
        except Exception:
            log.warning(
                "NLTK WordNet lemmatizáló nem elérhető — egyszerű tartalék "
                "lemmatizálót használok. Élesben telepítsd: pip install nltk"
            )
            _LEMMATIZER = _NaiveLemmatizer()
    return _LEMMATIZER


# ==========================================================================
# 2) CÍMKÉZÉS — pontszámból 3 osztály
# ==========================================================================
def score_to_sentiment(
    score: float,
    scale_max: float = 10.0,
    neg_threshold: float = CFG.neg_threshold,
    pos_threshold: float = CFG.pos_threshold,
) -> str | float:
    """
    Pontszám -> negative / neutral / positive.

    A küszöböket a 10-es skálán értelmezzük, és minden más skálát
    (pl. TripAdvisor 1–5) arra normalizálunk. Így UGYANAZ a címkézési
    szabály vonatkozik minden adathalmazra — ez alapfeltétele annak,
    hogy az RQ5 cross-dataset összehasonlítás értelmes legyen.
    """
    if pd.isna(score):
        return np.nan
    s = float(score) * (10.0 / float(scale_max))
    if s <= neg_threshold:
        return "negative"
    if s >= pos_threshold:
        return "positive"
    return "neutral"


def quantile_sentiment_labels(
    scores: pd.Series,
    q_neg: float = 0.25,
    q_pos: float = 0.60,
) -> tuple[pd.Series, float, float]:
    """
    ALTERNATÍV címkézés: a küszöböket nem kézzel adjuk meg, hanem az adat
    eloszlásából számoljuk (kvantilisek).

    Mikor van erre szükség?
    -----------------------
    Ha a pontszámok erősen ferdék. A Booking-adatban az átlag 8,87 és a medián
    9,0 — a vendégek túlnyomó többsége magas pontot ad. Egy fix "score ≤ 6 =
    negatív" szabály ilyenkor alig pár száz negatív példát ad, amiből a BERT
    nem tud tanulni.

    A kvantilis-alapú vágás garantálja a használható osztályméreteket:
        q_neg=0.25, q_pos=0.60  ->  kb. 25% / 35% / 40%

    ⚠ ÉRTELMEZÉSI KÜLÖNBSÉG, amit a dolgozatban le KELL írni:
      A fix küszöb ABSZOLÚT elégedettséget mér ("6 alatt = rossz élmény").
      A kvantilis-alapú RELATÍV elégedettséget ("az alsó negyed").
      Egy 8,0-s pontszám abszolút mércével jó, ehhez a mintához képest viszont
      az alsó negyedbe eshet. Mindkettő védhető, de nem ugyanaz az állítás —
      és a "negative" osztály jelentése is más lesz.

    Visszatér: (címkék, negatív küszöb, pozitív küszöb)
    """
    s = pd.to_numeric(scores, errors="coerce")
    t_neg = float(s.quantile(q_neg))
    t_pos = float(s.quantile(q_pos))

    if t_neg >= t_pos:
        raise ValueError(
            f"A két küszöb egybeesik (neg={t_neg}, pos={t_pos}). A pontszám "
            "valószínűleg túl kevés különböző értéket vesz fel ezekhez a "
            "kvantilisekhez. Válassz távolabbi q_neg / q_pos értékeket."
        )

    labels = pd.Series(
        np.where(s <= t_neg, "negative",
                 np.where(s >= t_pos, "positive", "neutral")),
        index=s.index,
    )
    labels[s.isna()] = np.nan
    return labels, t_neg, t_pos


def label_distribution_report(
    scores: pd.Series,
    scale_max: pd.Series | float = 10.0,
    schemes: list[tuple[str, float, float]] | None = None,
    quantile_schemes: list[tuple[str, float, float]] | None = None,
) -> pd.DataFrame:
    """
    Összehasonlító táblázat: hogyan alakul az osztályeloszlás különböző
    címkézési szabályokkal?

    Ez a táblázat a dolgozat 6.1 fejezetének egyik kulcsábrája: megmutatja,
    hogy a küszöbválasztás nem mellékes technikai részlet, hanem érdemben
    meghatározza a feladat nehézségét.

    Visszatér: soronként egy szabály, oszlopokban az osztályarányok, a
    legkisebb osztály mérete és a kiegyensúlyozatlansági arány.
    """
    schemes = schemes or [
        ("fix 6,0 / 8,0", 6.0, 8.0),
        ("fix 7,0 / 9,0", 7.0, 9.0),
        ("fix 7,5 / 9,0", 7.5, 9.0),
        ("fix 8,0 / 9,0", 8.0, 9.0),
        ("fix 8,0 / 9,5", 8.0, 9.5),
    ]
    quantile_schemes = quantile_schemes if quantile_schemes is not None else [
        ("kvantilis 25% / 60%", 0.25, 0.60),
        ("kvantilis 33% / 67%", 1 / 3, 2 / 3),
        ("kvantilis 20% / 55%", 0.20, 0.55),
    ]

    sm = (scale_max if isinstance(scale_max, pd.Series)
          else pd.Series([float(scale_max)] * len(scores), index=scores.index))

    rows = []

    def _row(name: str, labels: pd.Series, t_neg: float, t_pos: float):
        vc = labels.value_counts(normalize=True)
        counts = labels.value_counts()
        neg, neu, pos = (vc.get("negative", 0.0), vc.get("neutral", 0.0),
                         vc.get("positive", 0.0))
        parts = [p for p in (neg, neu, pos) if p > 0]
        rows.append({
            "szabály": name,
            "küszöbök": f"{t_neg:.2f} / {t_pos:.2f}",
            "negative": neg, "neutral": neu, "positive": pos,
            "legkisebb osztály (db)": int(counts.min()) if len(counts) else 0,
            "kiegyensúlyozatlanság": (max(parts) / min(parts)) if parts else np.nan,
        })

    for name, t_neg, t_pos in schemes:
        labels = pd.Series(
            [score_to_sentiment(s, m, neg_threshold=t_neg, pos_threshold=t_pos)
             for s, m in zip(scores, sm)],
            index=scores.index,
        ).dropna()
        _row(name, labels, t_neg, t_pos)

    for name, q_neg, q_pos in quantile_schemes:
        try:
            labels, t_neg, t_pos = quantile_sentiment_labels(scores, q_neg, q_pos)
            _row(name, labels.dropna(), t_neg, t_pos)
        except ValueError as e:
            log.warning("%s kihagyva: %s", name, e)

    return (pd.DataFrame(rows)
            .set_index("szabály")
            .round({"negative": 4, "neutral": 4, "positive": 4,
                    "kiegyensúlyozatlanság": 1}))


def truncation_report(
    df: pd.DataFrame,
    max_length: int,
    text_col: str = "text_bert",
    label_col: str = "sentiment",
    tokens_per_word: float = 1.3,
) -> pd.DataFrame:
    """
    Hány reviewt vág le a BERT `max_length` korlátja, és TORZÍT-E?

    Miért számít ez ennél az adatnál különösen?
    -------------------------------------------
    A szöveg felépítése:  cím . pozitív_rész   negatív_rész
    A tokenizáló a VÉGÉT vágja le — vagyis elsőként a NEGATÍV tartalmat.
    Ha ez sok reviewt érint, a modell szisztematikusan kevesebb negatív
    jelet lát, mint amennyi a szövegben van.

    Ez a függvény megmutatja, hogy a csonkolt reviewk osztályeloszlása
    eltér-e az egésztől. Ha igen, a max_length-et növelni kell.

    A becslés szóalapú (1 szó ≈ 1,3 token angolban), ami elég pontos ahhoz,
    hogy döntsünk. A 03-as notebook a valódi tokenizálóval is leméri.
    """
    words = df[text_col].fillna("").str.split().str.len()
    est_tokens = words * tokens_per_word
    trunc = est_tokens > (max_length - 2)      # [CLS] és [SEP] is elfér

    out = pd.DataFrame({
        "arány": [(~trunc).mean(), trunc.mean()],
        "db": [int((~trunc).sum()), int(trunc.sum())],
    }, index=["elfér", "csonkolva"])

    if label_col in df.columns:
        for label in df[label_col].astype(str).unique():
            mask = df[label_col].astype(str) == label
            out[label] = [
                (mask & ~trunc).sum() / max((~trunc).sum(), 1),
                (mask & trunc).sum() / max(trunc.sum(), 1),
            ]

    return out.round(4)


def add_sentiment_labels(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Hozzáadja a 'sentiment' oszlopot a 'score' és 'score_scale_max' alapján."""
    df = df.copy()
    df["sentiment"] = [
        score_to_sentiment(s, sm, **kwargs)
        for s, sm in zip(df["score"], df["score_scale_max"])
    ]
    df = df.dropna(subset=["sentiment"])
    # Kategorikus típus a CFG.labels sorrendjével -> konzisztens ábrák
    df["sentiment"] = pd.Categorical(df["sentiment"], categories=list(CFG.labels), ordered=True)
    return df


# ==========================================================================
# 3) SZŰRÉS / TISZTÍTÁS TÁBLASZINTEN
# ==========================================================================
def filter_reviews(
    df: pd.DataFrame,
    min_chars: int = CFG.min_chars,
    max_chars: int = CFG.max_chars,
    drop_duplicates: bool = CFG.drop_duplicates,
    keep_english_only: bool = CFG.keep_english_only,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sorszintű szűrés. A `verbose=True` kiírja, hány sort veszítettünk
    melyik lépésben — ezt a táblázatot TEDD BE A DOLGOZATBA, mert a
    bírálók szeretik látni az adattisztítás számszerű hatását.
    """
    steps: list[tuple[str, int]] = [("kiindulás", len(df))]
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str)

    df = df[df["text"].str.len() >= min_chars]
    steps.append((f"rövid szövegek (<{min_chars} kar.) eltávolítva", len(df)))

    df = df[df["text"].str.len() <= max_chars]
    steps.append((f"extrém hosszúak (>{max_chars} kar.) eltávolítva", len(df)))

    if drop_duplicates:
        df = df.drop_duplicates(subset=["text"])
        steps.append(("duplikátumok eltávolítva", len(df)))

    if keep_english_only:
        df = df[_is_probably_english(df["text"])]
        steps.append(("nem angol szövegek eltávolítva", len(df)))

    if verbose:
        prev = steps[0][1]
        log.info("--- Szűrési napló ---")
        for name, n in steps:
            log.info("%-45s %8d  (%+d)", name, n, n - prev)
            prev = n

    return df.reset_index(drop=True)


def _is_probably_english(texts: pd.Series, min_ratio: float = 0.90) -> pd.Series:
    """
    Gyors heurisztika: ha a BETŰK döntő többsége ASCII, valószínűleg angol.

    Miért csak a betűket nézzük?
    ----------------------------
    Az első változat az ÖSSZES karaktert számolta, és ez hibázott az
    emodzsikon. A hotel-reviewk tele vannak velük:

        "Very close proximity to the famous Florida street👌🏾👌🏾"

    Ez tiszta angol mondat, de a két emodzsi (egyenként több nem-ASCII
    kódpont) lenyomta az arányt, és a szűrő kidobta volna. Márpedig az
    emodzsi sentiment-hordozó jel — pont NEM akarjuk elveszíteni.

    Betűkre szűkítve az emodzsi, a számok és az írásjelek nem számítanak:
    a mérce az lesz, hogy a szöveg betűi latin ábécések-e. Egy görög vagy
    cirill betűs review továbbra is kiesik, egy emodzsis angol nem.

    (Ha a dolgozatban pontosabbat akarsz állítani, futtasd le a `langdetect`-et
    egy 1000-es mintán, és közöld, hogy mennyire egyezik ezzel a heurisztikával.)
    """
    def ratio(t: str) -> float:
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 1.0        # nincs betű — ezen a szűrőn ne bukjon el
        return sum(c.isascii() for c in letters) / len(letters)

    return texts.map(ratio) > min_ratio


def balance_classes(df: pd.DataFrame, strategy: str = "downsample",
                    seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    Osztály-kiegyensúlyozás.

    FONTOS: CSAK a TANÍTÓ halmazon szabad! A validációs és teszt halmaznak
    a valós eloszlást kell tükröznie, különben a mért pontosság hamis képet ad.
    """
    if strategy == "none":
        return df
    if strategy != "downsample":
        raise ValueError(f"Ismeretlen stratégia: {strategy}")
    n = df["sentiment"].value_counts().min()
    return (df.groupby("sentiment", observed=True, group_keys=False)
              .apply(lambda g: g.sample(n=n, random_state=seed))
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))


# ==========================================================================
# 4) ADATFELOSZTÁS 60 / 20 / 20
# ==========================================================================
def train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    train_size: float = CFG.train_size,
    val_size: float = CFG.val_size,
    seed: int = CFG.random_seed,
    stratify: bool = CFG.stratify,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Két lépésben osztunk: először leválasztjuk a tanítót, majd a maradékot
    felezzük validációra és tesztre.

    Rétegzett (stratified) felosztás: minden halmazban ugyanaz marad az
    osztályarány. 3 osztályos, kiegyensúlyozatlan adatnál ez kötelező,
    különben a teszthalmazba véletlenül kevés neutral kerülhet.
    """
    from sklearn.model_selection import train_test_split

    strat = df[label_col] if stratify else None
    train, rest = train_test_split(
        df, train_size=train_size, random_state=seed, stratify=strat, shuffle=True
    )
    # A maradékból mekkora rész legyen a validáció?
    rel_val = val_size / (1.0 - train_size)
    strat_rest = rest[label_col] if stratify else None
    val, test = train_test_split(
        rest, train_size=rel_val, random_state=seed, stratify=strat_rest, shuffle=True
    )

    log.info("Felosztás — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ==========================================================================
# 5) TELJES PIPELINE EGY HÍVÁSSAL
# ==========================================================================
def build_dataset(
    df_raw: pd.DataFrame,
    classic_clean: bool = True,
    minimal_clean: bool = True,
    balance: str = CFG.balance_strategy,
) -> pd.DataFrame:
    """
    Nyers tábla -> modellezésre kész tábla.

    Kimeneti oszlopok:
        review_id, text, text_clean (TF-IDF-hez), text_bert (BERT-hez),
        score, sentiment, dataset
    """
    df = filter_reviews(df_raw)
    df = add_sentiment_labels(df)

    if minimal_clean:
        df["text_bert"] = df["text"].map(clean_text_minimal)
    if classic_clean:
        log.info("Klasszikus tisztítás futtatása %d soron (eltarthat egy ideig)...", len(df))
        df["text_clean"] = df["text"].map(clean_text_classic)
        # Ha a tisztítás után üres lett, azt a sort eldobjuk
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    if balance != "none":
        df = balance_classes(df, balance)

    return df
