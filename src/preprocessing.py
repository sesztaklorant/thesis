"""Text cleaning, filtering, sentence and clause splitting, data splits.

Two cleaning paths are used: a full one for TF-IDF (lower case, punctuation, stop words,
lemmatisation) and a minimal one for BERT, which needs the original wording and the
negations. Sentiment labels are not derived from the score here; they come from
bert_labelling.py.
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
# 1) TEXT CLEANING
# ==========================================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_text_minimal(text: str) -> str:
    """
    Path (B) - for BERT. Only real noise is removed (HTML, URL, e-mail);
    punctuation, capitals, emojis and negations stay. The first step decodes
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
    Path (A) - for TF-IDF. Every step can be switched separately (for ablation).
    keep_negations=True: the stopword filter does not drop negation words.
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


# Lazy loading of NLTK resources (only when really needed).
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
    except Exception:  # noqa: BLE001
        # Fallback list if NLTK is unavailable.
        log.warning("NLTK stopword list not available, using a built-in minimal list.")
        words = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
                 "it", "we", "i", "for", "on", "with", "at", "this", "that"}
    if keep_negations:
        words = words - NEGATION_WORDS
    _STOPWORDS_CACHE[keep_negations] = words
    return words


class _NaiveLemmatizer:
    """
    Fallback lemmatiser if NLTK WordNet is unavailable (only the most common endings).
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
        except Exception:  # noqa: BLE001
            log.warning("NLTK WordNet not available — using the simple fallback lemmatiser.")
            _LEMMATIZER = _NaiveLemmatizer()
    return _LEMMATIZER


# ==========================================================================
# 2) SENTENCE SPLITTING
# ==========================================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 2) -> list[str]:
    """
    Simple sentence splitting, usually enough for short hotel reviews.
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
    Review-level table -> sentence-level table (review_id, sentence_idx, sentence).
    """
    out = df[[id_col, text_col]].copy()
    out["sentence"] = out[text_col].map(lambda t: split_sentences(t, min_words))
    out = out.drop(columns=[text_col]).explode("sentence").dropna(subset=["sentence"])
    out["sentence_idx"] = out.groupby(id_col).cumcount()
    return out.reset_index(drop=True)


# Clause boundaries: sentence end, line break, semicolon, and contrastive conjunctions
# ("but", "however", "although", "though", "except"). One clause thus typically holds
# one statement: "great location but the room was dirty" -> two units.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|\s*\n+\s*|\s*;\s*|,?\s+(?:but|however|although|though|except)\s+",
    flags=re.IGNORECASE,
)


def split_clauses(text: str, min_words: int = 2) -> list[str]:
    """
    Splitting into clauses (finer than split_sentences). Pieces shorter than min_words
    are dropped ("Nothing.", "Ok") - they contain no analysable content.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip(" ,.-") for p in _CLAUSE_SPLIT_RE.split(clean_text_minimal(text))]
    return [p for p in parts if len(p.split()) >= min_words]


def split_list_items(clause: str, min_words: int = 2) -> list[str]:
    """
    Splitting an enumeration at commas: "Great location, friendly staff, clean rooms" -> three
    units. Praise typically comes as such lists; left together, the topic model gives mixed
    "location / staff / clean" topics. A piece shorter than min_words sticks to the next one (the
    last to the previous one): "Clean, comfortable, great location" thus stays together.
    """
    pieces = [p.strip(" .-") for p in clause.split(",")]
    pieces = [p for p in pieces if p]
    out: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry}, {piece}" if carry else piece
        if len(piece.split()) < min_words:
            carry = piece
            continue
        out.append(piece)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]}, {carry}"
        else:
            out.append(carry)
    return out


def build_field_units(df: pd.DataFrame, fields: tuple[str, ...] = ("review_positive", "review_negative"),
                      id_col: str = "review_id", min_words: int = 2,
                      list_fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Analysis units per field: Booking's praise and complaint fields are split into clauses
    separately, so the two never merge into one "sentence" (this caused the mixed topics).
    field_polarity = which field it comes from (the guest's own classification).

    list_fields: in these fields enumerations are split too (split_list_items); by default the
    praise field, if CFG.split_praise_lists is true. The complaint field is not split at commas,
    because there a comma usually structures a single complaint.

    Returns: review_id | field | field_polarity | unit_idx | sentence
    """
    if list_fields is None:
        list_fields = ("review_positive",) if CFG.split_praise_lists else ()

    def units_of(text: str, split_lists: bool) -> list[str]:
        clauses = split_clauses(text, min_words)
        if not split_lists:
            return clauses
        return [item for c in clauses for item in split_list_items(c, min_words)]

    frames = []
    for field in fields:
        if field not in df.columns:
            log.warning("Field not found, skipped: %s", field)
            continue
        part = df[[id_col, field]].copy()
        part["sentence"] = part[field].map(lambda t, sl=field in list_fields: units_of(t, sl))
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
# 3) ROW-LEVEL FILTERING
# ==========================================================================
# --------------------------------------------------------------------------
# Non-complaint units: "nothing", "no complaints", "all was perfect" etc. written into the
# complaint field. They are not complaints, but because of the negation the model often
# reads them as negative and they form a separate "junk topic" in the subtopic model. The
# unit is cut into parts at commas/full stops; it is a non-complaint only if every part
# matches one of the patterns (so "nothing works" or "nothing to eat nearby" stays).
# --------------------------------------------------------------------------
_FILLER = r"(?:really|absolutely|honestly|literally|just|at all|whatsoever|in particular|so far|to be honest|tbh)"
_NON_COMPLAINT_PARTS = [
    rf"(?:there (?:was|is|were)\s+|there'?s\s+)?(?:{_FILLER}\s+)*(?:nothing|none|nil|nope|no|n/?a|na|-+)(?:\s+{_FILLER})*",
    rf"nothing\s+(?:{_FILLER}\s+)*(?:to (?:complain|report|mention|say|dislike|add|comment)(?: about| on)?|bad|negative|wrong|special)(?:\s+{_FILLER})*",
    r"(?:there (?:was|is) |there'?s )?(?:really |absolutely )?nothing (?:that )?(?:i |we )?(?:did ?n'?t|do ?n'?t|did not|do not|could ?n'?t) (?:like|liked|enjoy|love)(?: about it)?",
    r"(?:i |we )?(?:can'?t|cannot|could ?n'?t|could not) (?:think of|find|say|name) anything(?: bad| negative| wrong| to complain about)?",
    rf"no (?:complaints?|negatives?|cons|issues?|problems?|dislikes?|comments?)(?:\s+{_FILLER})*",
    r"(?:everything|all|it|all of it) (?:was |is |were )?(?:just |really |absolutely )?(?:good|great|perfect|fine|ok|okay|excellent|amazing|wonderful|lovely|brilliant|superb|fantastic|as expected)",
    r"(?:i |we )?(?:liked|loved|enjoyed) (?:everything|it all|all of it)",
    r"not applicable|nothing comes to mind|all good|all perfect|everything ok",
]
_NON_COMPLAINT_RE = re.compile(r"^(?:" + "|".join(_NON_COMPLAINT_PARTS) + r")$", re.IGNORECASE)
# "...not the hotel's fault" - the guest says it is not the property's fault (not an actionable complaint).
_NOT_THEIR_FAULT_RE = re.compile(
    r"\b(?:not|no|nothing|wasn'?t|isn'?t|was not|is not)\b[^.]{0,25}\b(?:hotel|hotels|property|host|hosts|staff|"
    r"owner|owners|apartment|place)'?s?\b[^.]{0,10}\bfault\b", re.IGNORECASE)


def is_non_complaint(text: str) -> bool:
    """True if the unit is not a real complaint (see above)."""
    if not isinstance(text, str):
        return True
    norm = re.sub(r"[!?:\"()*~]+", " ", text.lower().replace("’", "'"))
    if _NOT_THEIR_FAULT_RE.search(norm):
        return True
    parts = [re.sub(r"\s+", " ", p).strip(" -") for p in re.split(r"[,.;]+", norm)]
    parts = [p for p in parts if p]
    return bool(parts) and all(_NON_COMPLAINT_RE.match(p) for p in parts)


def non_complaint_mask(texts: pd.Series) -> pd.Series:
    """Boolean mask for a text column."""
    return texts.map(is_non_complaint).astype(bool)


def filter_reviews(
    df: pd.DataFrame,
    min_chars: int = CFG.min_chars,
    max_chars: int = CFG.max_chars,
    drop_duplicates: bool = CFG.drop_duplicates,
    keep_english_only: bool = CFG.keep_english_only,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Row-level filtering. Returns the filtered table and the filtering log (how
    many rows each step removed).
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
    Quick heuristic: at least 90% of the letters are ASCII. Only letters are
    considered, so emojis (which carry sentiment) do not cause a text to be dropped.
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
    Estimate what share of reviews BERT's max_length limit truncates (in English
    1 word ≈ 1.3 tokens). Notebook 02 also measures it with the real tokenizer.
    """
    est_tokens = df[text_col].fillna("").str.split().str.len() * tokens_per_word
    truncated = est_tokens > (max_length - 2)  # [CLS] + [SEP]
    return pd.DataFrame({
        "share": [(~truncated).mean(), truncated.mean()],
        "count": [int((~truncated).sum()), int(truncated.sum())],
    }, index=["fits", "truncated"]).round(4)


# ==========================================================================
# 4) THE FULL PREPARATION
# ==========================================================================
def build_dataset(df_raw: pd.DataFrame, classic_clean: bool = True,
                  minimal_clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Raw table -> prepared table (without labels).
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
# 5) SPLIT FOR THE CLASSICAL MODELS
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
    Stratified 60/20/20 split (class proportions are equal in every part).
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
    At most max_per_class examples per class (None = size of the smallest class).
    only on a training set; validation and test keep the real distribution.
    """
    counts = df[label_col].value_counts()
    cap = int(counts.min()) if max_per_class is None else int(max_per_class)
    parts = [group.sample(n=min(len(group), cap), random_state=seed)
             for _, group in df.groupby(label_col, observed=True)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
