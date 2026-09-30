"""
preprocessing.py — Text cleaning, labelling, data splitting.

Chapter 6.1 of the thesis ("Data Preprocessing") is written from this module.

A key design decision that has to be described in the thesis as well:
---------------------------------------------------------------------
There are TWO DIFFERENT preprocessing paths, and this is INTENTIONAL:

  (A) Classical ML (TF-IDF + Naive Bayes / LogReg / SVM):
      aggressive normalisation — lowercasing, punctuation removal,
      stopword filtering, lemmatisation. TF-IDF is a "bag-of-words" model,
      it does not understand word order, so reducing noise helps.

  (B) Transformer (BERT):
      MINIMAL cleaning. BERT uses its own tokenizer and learns from context.
      If you remove the stopwords and the negations ("not clean" -> "clean"),
      you destroy the meaning and get WORSE results. This is a common
      beginner's mistake — and exactly for that reason good material for
      answering RQ2.
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
# 1) TEXT CLEANING
# ==========================================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_text_minimal(text: str) -> str:
    """
    Path (B) — for BERT. Only real noise is removed.
    Punctuation, capital letters, emojis and negation words STAY.

    Decoding HTML entities
    ----------------------
    In the Booking.com export the text contains HTML-encoded apostrophes:

        "wouldn&#39;t suggest walking"   ->   "wouldn't suggest walking"

    This is NOT a cosmetic issue. If it is not decoded:
      • on the TF-IDF path the `[^a-z\\s']` filter throws out the characters
        of `&#39;`, and "wouldn&#39;t" becomes "wouldn t" — i.e. THE NEGATION
        IS LOST, which is the worst thing that can happen in sentiment analysis;
      • the BERT tokenizer splits the entity into separate tokens and learns
        meaningless fragments.

    `html.unescape` handles all common entities (`&#39;`, `&amp;`, `&quot;`,
    `&lt;`, `&nbsp;` etc.), not only the apostrophe.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)              # &#39; -> '   (THIS IS THE FIRST STEP)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)          # real HTML tags: <br>, <p>
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
    Path (A) — for TF-IDF. Every step can be switched separately, so that an
    ABLATION study can be run for RQ2:
    "how much does it matter if I switch off lemmatisation?"

    keep_negations: if True, the negation words (not, no, never, ...) are NOT
        thrown out by the stopword filter. For a sentiment task this usually
        helps — and it is a nice little experiment for the thesis on its own.
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


# --- Lazy loading of NLTK resources (only when really needed) ---
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
        log.warning("NLTK stopword list not available, using a built-in minimal list.")
        sw = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
              "it", "we", "i", "for", "on", "with", "at", "this", "that"}
    if keep_negations:
        sw = sw - NEGATION_WORDS
    _STOPWORDS_CACHE[keep_negations] = sw
    return sw


class _NaiveLemmatizer:
    """
    Fallback if NLTK/WordNet is not available (e.g. on an offline machine).
    Handles only the most common English plural and verb endings.

    It is IMPORTANT to mention in the thesis if this was used instead of
    NLTK — but in the normal case the NLTK WordNetLemmatizer runs.
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
            _LEMMATIZER.lemmatize("tests")  # check: does it really work?
        except Exception:
            log.warning(
                "NLTK WordNet lemmatizer not available — using a simple fallback "
                "lemmatizer. For real runs install it: pip install nltk"
            )
            _LEMMATIZER = _NaiveLemmatizer()
    return _LEMMATIZER


# ==========================================================================
# 2) LABELLING — 3 classes from the score
# ==========================================================================
def score_to_sentiment(
    score: float,
    scale_max: float = 10.0,
    neg_threshold: float = CFG.neg_threshold,
    pos_threshold: float = CFG.pos_threshold,
) -> str | float:
    """
    Score -> negative / neutral / positive.

    The thresholds are interpreted on the 10-point scale, and every other
    scale (e.g. TripAdvisor 1–5) is normalised to it. This way the SAME
    labelling rule applies to every dataset — a precondition for the RQ5
    cross-dataset comparison to be meaningful.
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
    ALTERNATIVE labelling: the thresholds are not set by hand but computed
    from the distribution of the data (quantiles).

    When is this needed?
    --------------------
    When the scores are strongly skewed. In the Booking data the mean is 8.87
    and the median 9.0 — the vast majority of guests give high scores. A fixed
    "score ≤ 6 = negative" rule then yields only a few hundred negative
    examples, which BERT cannot learn from.

    The quantile-based cut guarantees usable class sizes:
        q_neg=0.25, q_pos=0.60  ->  about 25% / 35% / 40%

    ⚠ DIFFERENCE IN INTERPRETATION that MUST be described in the thesis:
      The fixed threshold measures ABSOLUTE satisfaction ("below 6 = bad
      experience"). The quantile-based one measures RELATIVE satisfaction
      ("the bottom quarter"). A score of 8.0 is good on an absolute scale,
      but relative to this sample it may fall into the bottom quarter. Both
      are defensible, but they are not the same claim — and the meaning of
      the "negative" class differs too.

    Returns: (labels, negative threshold, positive threshold)
    """
    s = pd.to_numeric(scores, errors="coerce")
    t_neg = float(s.quantile(q_neg))
    t_pos = float(s.quantile(q_pos))

    if t_neg >= t_pos:
        raise ValueError(
            f"The two thresholds coincide (neg={t_neg}, pos={t_pos}). The score "
            "probably takes too few distinct values for these quantiles. "
            "Choose q_neg / q_pos values further apart."
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
    Comparison table: how does the class distribution change under different
    labelling rules?

    This table is one of the key exhibits of chapter 6.1 of the thesis: it
    shows that the choice of threshold is not a minor technical detail but
    substantially determines the difficulty of the task.

    Returns: one rule per row, with the class proportions, the size of the
    smallest class and the imbalance ratio in the columns.
    """
    # In practice the Booking score takes INTEGER values (3–10), so only those
    # threshold pairs differ from each other that put different integers into
    # different classes. These are the four meaningful splits:
    schemes = schemes or [
        ("A) 6 / 8   — strict",       6.0,  8.0),
        ("B) 7 / 9   — balanced",     7.0,  9.0),
        ("C) 7 / 10  — wide middle",  7.0, 10.0),
        ("D) 8 / 10  — relative",     8.0, 10.0),
    ]
    quantile_schemes = quantile_schemes if quantile_schemes is not None else []

    sm = (scale_max if isinstance(scale_max, pd.Series)
          else pd.Series([float(scale_max)] * len(scores), index=scores.index))

    # Is the score discrete? If so, it is worth showing WHICH values go into
    # which class — without this the table can be misleading (two different
    # threshold pairs can give exactly the same result).
    uniq = np.sort(pd.to_numeric(scores, errors="coerce").dropna().unique())
    main_vals = [v for v in uniq if float(v).is_integer()]
    discrete = len(uniq) <= 25
    if discrete:
        log.info("The score is discrete: %d distinct values, %d of them integers. "
                 "Only thresholds on integer boundaries give a different split.",
                 len(uniq), len(main_vals))

    rows = []

    def _row(name: str, labels: pd.Series, t_neg: float, t_pos: float):
        vc = labels.value_counts(normalize=True)
        counts = labels.value_counts()
        neg, neu, pos = (vc.get("negative", 0.0), vc.get("neutral", 0.0),
                         vc.get("positive", 0.0))
        parts = [p for p in (neg, neu, pos) if p > 0]

        row = {
            "scheme": name,
            "thresholds": f"{t_neg:g} / {t_pos:g}",
        }
        if discrete:
            def grp(lo_ok, hi_ok):
                xs = [int(v) for v in main_vals if lo_ok(v) and hi_ok(v)]
                return "–".join([str(xs[0]), str(xs[-1])]) if len(xs) > 1 else (
                    str(xs[0]) if xs else "empty!")
            row["neg scores"] = grp(lambda v: True, lambda v: v <= t_neg)
            row["neu scores"] = grp(lambda v: v > t_neg, lambda v: v < t_pos)
            row["pos scores"] = grp(lambda v: v >= t_pos, lambda v: True)

        row.update({
            "negative": neg, "neutral": neu, "positive": pos,
            "smallest class (n)": int(counts.min()) if len(counts) else 0,
            "imbalance ratio": (max(parts) / min(parts)) if parts else np.nan,
        })
        rows.append(row)

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
            log.warning("%s skipped: %s", name, e)

    return (pd.DataFrame(rows)
            .set_index("scheme")
            .round({"negative": 4, "neutral": 4, "positive": 4,
                    "imbalance ratio": 1}))


def truncation_report(
    df: pd.DataFrame,
    max_length: int,
    text_col: str = "text_bert",
    label_col: str = "sentiment",
    tokens_per_word: float = 1.3,
) -> pd.DataFrame:
    """
    How many reviews does BERT's `max_length` limit cut off, and does it BIAS
    the data?

    Why does this matter especially for this data?
    ----------------------------------------------
    The text is built as:  title . positive_part   negative_part
    The tokenizer cuts off the END — i.e. the NEGATIVE content goes first.
    If this affects many reviews, the model systematically sees fewer
    negative signals than the text contains.

    This function shows whether the class distribution of the truncated
    reviews differs from the whole. If it does, max_length has to be raised.

    The estimate is word-based (1 word ≈ 1.3 tokens in English), which is
    accurate enough for the decision. Notebook 03 also measures it with the
    real tokenizer.
    """
    words = df[text_col].fillna("").str.split().str.len()
    est_tokens = words * tokens_per_word
    trunc = est_tokens > (max_length - 2)      # [CLS] and [SEP] also have to fit

    out = pd.DataFrame({
        "share": [(~trunc).mean(), trunc.mean()],
        "count": [int((~trunc).sum()), int(trunc.sum())],
    }, index=["fits", "truncated"])

    if label_col in df.columns:
        for label in df[label_col].astype(str).unique():
            mask = df[label_col].astype(str) == label
            out[label] = [
                (mask & ~trunc).sum() / max((~trunc).sum(), 1),
                (mask & trunc).sum() / max(trunc.sum(), 1),
            ]

    return out.round(4)


def add_sentiment_labels(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add the 'sentiment' column based on 'score' and 'score_scale_max'."""
    df = df.copy()
    df["sentiment"] = [
        score_to_sentiment(s, sm, **kwargs)
        for s, sm in zip(df["score"], df["score_scale_max"])
    ]
    df = df.dropna(subset=["sentiment"])
    # Categorical type in CFG.labels order -> consistent figures
    df["sentiment"] = pd.Categorical(df["sentiment"], categories=list(CFG.labels), ordered=True)
    return df


# ==========================================================================
# 3) TABLE-LEVEL FILTERING / CLEANING
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
    Row-level filtering. `verbose=True` prints how many rows were lost in
    which step — PUT THIS TABLE INTO THE THESIS, reviewers like to see the
    quantitative effect of the data cleaning.
    """
    steps: list[tuple[str, int]] = [("start", len(df))]
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str)

    df = df[df["text"].str.len() >= min_chars]
    steps.append((f"short texts (<{min_chars} chars) removed", len(df)))

    df = df[df["text"].str.len() <= max_chars]
    steps.append((f"extremely long texts (>{max_chars} chars) removed", len(df)))

    if drop_duplicates:
        df = df.drop_duplicates(subset=["text"])
        steps.append(("duplicates removed", len(df)))

    if keep_english_only:
        df = df[_is_probably_english(df["text"])]
        steps.append(("non-English texts removed", len(df)))

    if verbose:
        prev = steps[0][1]
        log.info("--- Filtering log ---")
        for name, n in steps:
            log.info("%-45s %8d  (%+d)", name, n, n - prev)
            prev = n

    return df.reset_index(drop=True)


def _is_probably_english(texts: pd.Series, min_ratio: float = 0.90) -> pd.Series:
    """
    Quick heuristic: if the vast majority of the LETTERS are ASCII, it is
    probably English.

    Why look only at letters?
    -------------------------
    The first version counted ALL characters, and it failed on emojis.
    Hotel reviews are full of them:

        "Very close proximity to the famous Florida street👌🏾👌🏾"

    This is a plain English sentence, but the two emojis (each several
    non-ASCII code points) pushed the ratio down, and the filter would have
    thrown it out. Yet an emoji is a sentiment-bearing signal — exactly what
    we do NOT want to lose.

    Restricted to letters, emojis, digits and punctuation do not count: the
    criterion becomes whether the letters of the text are Latin. A review in
    Greek or Cyrillic script is still dropped, an English one with emojis is not.

    (If you want to make a more precise claim in the thesis, run `langdetect`
    on a sample of 1000 and report how well it agrees with this heuristic.)
    """
    def ratio(t: str) -> float:
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 1.0        # no letters — it should not fail on this filter
        return sum(c.isascii() for c in letters) / len(letters)

    return texts.map(ratio) > min_ratio


def balance_classes(df: pd.DataFrame, strategy: str = "downsample",
                    seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    Class balancing.

    IMPORTANT: ONLY on the TRAINING set! The validation and test sets must
    reflect the real distribution, otherwise the measured accuracy is false.
    """
    if strategy == "none":
        return df
    if strategy != "downsample":
        raise ValueError(f"Unknown strategy: {strategy}")
    n = df["sentiment"].value_counts().min()
    # groupby().sample() and NOT groupby().apply(): from pandas 3.0 apply
    # drops the grouping column, and "sentiment" would disappear.
    return (df.groupby("sentiment", observed=True)
              .sample(n=n, random_state=seed)
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))


# ==========================================================================
# 4) DATA SPLIT 60 / 20 / 20
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
    The split is done in two steps: first the training set is separated,
    then the rest is halved into validation and test.

    Stratified split: the class proportions stay the same in every set.
    For imbalanced 3-class data this is mandatory, otherwise the test set
    may by chance get too few neutral examples.
    """
    from sklearn.model_selection import train_test_split

    strat = df[label_col] if stratify else None
    train, rest = train_test_split(
        df, train_size=train_size, random_state=seed, stratify=strat, shuffle=True
    )
    # What share of the rest should be the validation set?
    rel_val = val_size / (1.0 - train_size)
    strat_rest = rest[label_col] if stratify else None
    val, test = train_test_split(
        rest, train_size=rel_val, random_state=seed, stratify=strat_rest, shuffle=True
    )

    log.info("Split — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ==========================================================================
# 5) THE FULL PIPELINE IN ONE CALL
# ==========================================================================
def build_dataset(
    df_raw: pd.DataFrame,
    classic_clean: bool = True,
    minimal_clean: bool = True,
    balance: str = CFG.balance_strategy,
) -> pd.DataFrame:
    """
    Raw table -> table ready for modelling.

    Output columns:
        review_id, text, text_clean (for TF-IDF), text_bert (for BERT),
        score, sentiment, dataset
    """
    df = filter_reviews(df_raw)
    df = add_sentiment_labels(df)

    if minimal_clean:
        df["text_bert"] = df["text"].map(clean_text_minimal)
    if classic_clean:
        log.info("Running classical cleaning on %d rows (this may take a while)...", len(df))
        df["text_clean"] = df["text"].map(clean_text_classic)
        # Rows that became empty after cleaning are dropped
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    if balance != "none":
        df = balance_classes(df, balance)

    return df
