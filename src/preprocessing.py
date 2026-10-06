"""
preprocessing.py — Szövegtisztítás, szűrés, mondatokra bontás, felosztás.
                   Text cleaning, filtering, sentence splitting, data split.

HU: Két SZÁNDÉKOSAN eltérő tisztítási út:
    (A) TF-IDF-hez: agresszív normalizálás (kisbetű, írásjelek, stopszavak,
        lemmatizálás) — a szózsák-modellnek a zajcsökkentés segít.
    (B) BERT-hez: MINIMÁLIS tisztítás. A transzformer a kontextusból tanul; ha
        a tagadást kivennénk ("not clean" -> "clean"), az ellenkezőjét tanítanánk.
EN: Two INTENTIONALLY different cleaning paths:
    (A) For TF-IDF: aggressive normalisation (lowercase, punctuation, stopwords,
        lemmatisation) — noise reduction helps the bag-of-words model.
    (B) For BERT: MINIMAL cleaning. The transformer learns from context; removing
        the negation ("not clean" -> "clean") would teach the opposite.

HU: FONTOS VÁLTOZÁS: ez a modul már NEM készít szentiment-címkét a pontszámból.
    A címkéket a bert_labelling.py állítja elő a SZÖVEGBŐL.
EN: IMPORTANT CHANGE: this module no longer derives sentiment labels from the
    score. Labels are produced from the TEXT by bert_labelling.py.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

import pandas as pd

from .config import CFG

log = logging.getLogger(__name__)


# ==========================================================================
# 1) SZÖVEGTISZTÍTÁS / TEXT CLEANING
# ==========================================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_text_minimal(text: str) -> str:
    """
    HU: (B) út — BERT-hez. Csak a valódi zaj megy ki (HTML, URL, e-mail);
        írásjel, nagybetű, emoji és tagadás MARAD. Első lépés a HTML-entitások
        dekódolása (wouldn&#39;t -> wouldn't), különben a tagadás elveszne.
    EN: Path (B) — for BERT. Only real noise is removed (HTML, URL, e-mail);
        punctuation, capitals, emojis and negations STAY. The first step decodes
        HTML entities (wouldn&#39;t -> wouldn't), otherwise the negation is lost.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)
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
    HU: (A) út — TF-IDF-hez. Minden lépés külön kapcsolható (ablációs vizsgálathoz).
        keep_negations=True: a tagadószavakat (not, no, never...) a stopszó-szűrő
        NEM dobja ki.
    EN: Path (A) — for TF-IDF. Every step can be switched separately (for ablation).
        keep_negations=True: the stopword filter does NOT drop negation words.
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
        stopwords = _get_stopwords(keep_negations) if remove_stopwords else set()
        lemmatizer = _get_lemmatizer() if lemmatize else None
        tokens = [lemmatizer.lemmatize(t) if lemmatizer else t
                  for t in tokens if t not in stopwords]
    return " ".join(tokens)


# HU: NLTK-erőforrások lusta betöltése (csak ha tényleg kell).
# EN: Lazy loading of NLTK resources (only when really needed).
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
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:                                             # noqa: BLE001
        # HU: Tartalék lista, ha az NLTK nem érhető el (ezt a dolgozatban jelezni kell).
        # EN: Fallback list if NLTK is unavailable (to be stated in the thesis).
        log.warning("NLTK stopword list not available, using a built-in minimal list.")
        words = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
                 "it", "we", "i", "for", "on", "with", "at", "this", "that"}
    if keep_negations:
        words = words - NEGATION_WORDS
    _STOPWORDS_CACHE[keep_negations] = words
    return words


class _NaiveLemmatizer:
    """
    HU: Tartalék lemmatizáló, ha az NLTK WordNet nem érhető el (csak a leggyakoribb végződések).
    EN: Fallback lemmatiser if NLTK WordNet is unavailable (only the most common endings).
    """

    def lemmatize(self, word: str) -> str:
        for suffix, replacement in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                                    ("shes", "sh"), ("xes", "x")):
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[: -len(suffix)] + replacement
        if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
            return word[:-1]
        return word


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
            _LEMMATIZER.lemmatize("tests")
        except Exception:                                         # noqa: BLE001
            log.warning("NLTK WordNet not available — using the simple fallback lemmatiser.")
            _LEMMATIZER = _NaiveLemmatizer()
    return _LEMMATIZER


# ==========================================================================
# 2) MONDATOKRA BONTÁS / SENTENCE SPLITTING
# ==========================================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 2) -> list[str]:
    """
    HU: Egyszerű mondatbontás. Hotelvéleményekre jellemzően elég, mert rövidek.
        Ha egy szöveg egyetlen mondatot sem ad (pl. írásjel nélküli rövid szöveg),
        az egész szöveget adja vissza, így egyetlen vélemény sem vész el.
    EN: Simple sentence splitting, usually enough for short hotel reviews.
        If a text yields no sentence (e.g. short text without punctuation), the
        whole text is returned, so no review is lost.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    sentences = [p for p in parts if len(p.split()) >= min_words]
    return sentences or [text.strip()]


def explode_to_sentences(df: pd.DataFrame, text_col: str = "text_bert",
                         id_col: str = "review_id", min_words: int = 2) -> pd.DataFrame:
    """
    HU: Véleményszintű tábla -> mondatszintű tábla (review_id, sentence_idx, sentence).
    EN: Review-level table -> sentence-level table (review_id, sentence_idx, sentence).
    """
    out = df[[id_col, text_col]].copy()
    out["sentence"] = out[text_col].map(lambda t: split_sentences(t, min_words))
    out = out.drop(columns=[text_col]).explode("sentence").dropna(subset=["sentence"])
    out["sentence_idx"] = out.groupby(id_col).cumcount()
    return out.reset_index(drop=True)


# HU: Tagmondat-határok: mondatvég, sortörés, pontosvessző, és az ellentétes kötőszavak
#     ("but", "however", "although", "though", "except"). Egy tagmondatba így jellemzően
#     EGY állítás kerül: "great location but the room was dirty" -> két egység.
# EN: Clause boundaries: sentence end, line break, semicolon, and contrastive conjunctions
#     ("but", "however", "although", "though", "except"). One clause thus typically holds
#     ONE statement: "great location but the room was dirty" -> two units.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|\s*\n+\s*|\s*;\s*|,?\s+(?:but|however|although|though|except)\s+",
    flags=re.IGNORECASE,
)


def split_clauses(text: str, min_words: int = 2) -> list[str]:
    """
    HU: Tagmondatokra bontás (finomabb, mint a split_sentences). A min_words-nél rövidebb
        darabok kimaradnak ("Nothing.", "Ok") — ezekben nincs elemezhető tartalom.
    EN: Splitting into clauses (finer than split_sentences). Pieces shorter than min_words
        are dropped ("Nothing.", "Ok") — they contain no analysable content.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip(" ,.-") for p in _CLAUSE_SPLIT_RE.split(clean_text_minimal(text))]
    return [p for p in parts if len(p.split()) >= min_words]


def build_field_units(df: pd.DataFrame, fields: tuple[str, ...] = ("review_positive", "review_negative"),
                      id_col: str = "review_id", min_words: int = 2) -> pd.DataFrame:
    """
    HU: Elemzési egységek MEZŐNKÉNT: a Booking dicséret- és panaszmezőjét külön bontjuk
        tagmondatokra, így a kettő soha nem olvad egy "mondatba" (ez okozta a vegyes témákat).
        field_polarity = melyik mezőből jön (a vendég saját besorolása).
    EN: Analysis units PER FIELD: Booking's praise and complaint fields are split into clauses
        separately, so the two never merge into one "sentence" (this caused the mixed topics).
        field_polarity = which field it comes from (the guest's own classification).

    Returns: review_id | field | field_polarity | unit_idx | sentence
    """
    frames = []
    for field in fields:
        if field not in df.columns:
            log.warning("Field not found, skipped: %s", field)
            continue
        part = df[[id_col, field]].copy()
        part["sentence"] = part[field].map(lambda t: split_clauses(t, min_words))
        part = part.drop(columns=[field]).explode("sentence").dropna(subset=["sentence"])
        part["field"] = field
        part["field_polarity"] = "positive" if "positive" in field else "negative"
        frames.append(part)
    units = pd.concat(frames, ignore_index=True)
    units["unit_idx"] = units.groupby([id_col, "field"]).cumcount()
    log.info("Field units: %d from %d reviews | by field: %s", len(units), units[id_col].nunique(),
             units["field_polarity"].value_counts().to_dict())
    return units[[id_col, "field", "field_polarity", "unit_idx", "sentence"]].reset_index(drop=True)


# ==========================================================================
# 3) SOR-SZINTŰ SZŰRÉS / ROW-LEVEL FILTERING
# ==========================================================================
def filter_reviews(
    df: pd.DataFrame,
    min_chars: int = CFG.min_chars,
    max_chars: int = CFG.max_chars,
    drop_duplicates: bool = CFG.drop_duplicates,
    keep_english_only: bool = CFG.keep_english_only,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    HU: Sor-szintű szűrés. Visszaadja a szűrt táblát ÉS a szűrési naplót (melyik
        lépés hány sort vitt el) — ez a napló táblázatként a dolgozatba kerül.
    EN: Row-level filtering. Returns the filtered table AND the filtering log (how
        many rows each step removed) — this log goes into the thesis as a table.
    """
    steps: list[tuple[str, int]] = [("start", len(df))]
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    df = df[df["text"].str.len() >= min_chars]
    steps.append((f"short texts (<{min_chars} chars) removed", len(df)))

    df = df[df["text"].str.len() <= max_chars]
    steps.append((f"very long texts (>{max_chars} chars) removed", len(df)))

    if drop_duplicates:
        df = df.drop_duplicates(subset=["text"])
        steps.append(("duplicates removed", len(df)))

    if keep_english_only:
        df = df[is_probably_english(df["text"])]
        steps.append(("non-English texts removed", len(df)))

    log_df = pd.DataFrame(steps, columns=["step", "rows"])
    log_df["change"] = log_df["rows"].diff().fillna(0).astype(int)
    if verbose:
        log.info("--- Filtering log ---\n%s", log_df.to_string(index=False))
    return df.reset_index(drop=True), log_df


def is_probably_english(texts: pd.Series, min_ratio: float = 0.90) -> pd.Series:
    """
    HU: Gyors heurisztika: a BETŰK legalább 90%-a ASCII. Csak a betűket nézzük,
        így az emojik (amelyek szentimentet hordoznak) nem dobják ki a szöveget.
    EN: Quick heuristic: at least 90% of the LETTERS are ASCII. Only letters are
        considered, so emojis (which carry sentiment) do not drop the text.
    """
    def ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 1.0
        return sum(c.isascii() for c in letters) / len(letters)

    return texts.map(ratio) > min_ratio


def truncation_report(df: pd.DataFrame, max_length: int, text_col: str = "text_bert",
                      tokens_per_word: float = 1.3) -> pd.DataFrame:
    """
    HU: Becslés: a vélemények hány százalékát vágja le a BERT max_length korlátja
        (angolban 1 szó ≈ 1,3 token). A 02-es notebook a valódi tokenizálóval is méri.
    EN: Estimate what share of reviews BERT's max_length limit truncates (in English
        1 word ≈ 1.3 tokens). Notebook 02 also measures it with the real tokenizer.
    """
    est_tokens = df[text_col].fillna("").str.split().str.len() * tokens_per_word
    truncated = est_tokens > (max_length - 2)        # HU/EN: [CLS] + [SEP]
    return pd.DataFrame({
        "share": [(~truncated).mean(), truncated.mean()],
        "count": [int((~truncated).sum()), int(truncated.sum())],
    }, index=["fits", "truncated"]).round(4)


# ==========================================================================
# 4) A TELJES ELŐKÉSZÍTÉS / THE FULL PREPARATION
# ==========================================================================
def build_dataset(df_raw: pd.DataFrame, classic_clean: bool = True,
                  minimal_clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    HU: Nyers tábla -> előkészített tábla (címkék NÉLKÜL).
        Új oszlopok: text_bert (BERT-hez), text_clean (TF-IDF-hez), n_words, n_chars.
    EN: Raw table -> prepared table (WITHOUT labels).
        New columns: text_bert (for BERT), text_clean (for TF-IDF), n_words, n_chars.

    Returns: (prepared table, filtering log)
    """
    df, filter_log = filter_reviews(df_raw)
    if minimal_clean:
        df["text_bert"] = df["text"].map(clean_text_minimal)
    if classic_clean:
        log.info("Classical cleaning on %d rows (lemmatisation is the slow part)...", len(df))
        df["text_clean"] = df["text"].map(clean_text_classic)
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    return df, filter_log


# ==========================================================================
# 5) FELOSZTÁS A KLASSZIKUS MODELLEKHEZ / SPLIT FOR THE CLASSICAL MODELS
# ==========================================================================
def train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "label_final",
    train_size: float = CFG.train_size,
    val_size: float = CFG.val_size,
    seed: int = CFG.random_seed,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    HU: Rétegzett 60/20/20 felosztás (az osztályarányok minden részben azonosak).
    EN: Stratified 60/20/20 split (class proportions are equal in every part).
    """
    from sklearn.model_selection import train_test_split

    strat = df[label_col] if stratify else None
    train, rest = train_test_split(df, train_size=train_size, random_state=seed,
                                   stratify=strat, shuffle=True)
    relative_val = val_size / (1.0 - train_size)
    strat_rest = rest[label_col] if stratify else None
    val, test = train_test_split(rest, train_size=relative_val, random_state=seed,
                                 stratify=strat_rest, shuffle=True)
    log.info("Split — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def balance_classes(df: pd.DataFrame, label_col: str, max_per_class: int | None = None,
                    seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Osztályonként legfeljebb max_per_class példa (None = a legkisebb osztály mérete).
        CSAK tanítóhalmazon! A validációs/teszthalmaz a valós eloszlást tükrözze.
    EN: At most max_per_class examples per class (None = size of the smallest class).
        ONLY on a training set! Validation/test must reflect the real distribution.
    """
    counts = df[label_col].value_counts()
    cap = int(counts.min()) if max_per_class is None else int(max_per_class)
    parts = [group.sample(n=min(len(group), cap), random_state=seed)
             for _, group in df.groupby(label_col, observed=True)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
