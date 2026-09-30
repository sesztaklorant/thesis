"""
aspects.py — Aspect-based sentiment analysis (ABSA), RQ3 and RQ4.

There are two levels, and it is worth structuring the thesis this way too:

  1. LEXICON-BASED ASPECT DETECTION (in this file)
     Keyword lists per aspect + sentence splitting. Simple, fast,
     transparent, and gives numbers for RQ3 immediately ("which aspects are
     most often tied to negative sentiment?").
     This is the BASELINE the transformer model is compared to.

  2. TRANSFORMER-BASED ABSA (in notebook 04, trained on HRAST)
     The aspect and the sentence are given to BERT as a pair:
         [CLS] room [SEP] the room was extremely hot [SEP] -> negative
     This "sentence-pair" format has been standard since SemEval-2014.

Comparing the two is good thesis material in itself: you show where the
keyword approach fails (negation, irony, implicit aspects) and how much
better the transformer handles it.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .config import CFG

log = logging.getLogger(__name__)


# ==========================================================================
# 1) ASPECT LEXICON
# ==========================================================================
# IMPORTANT: this list has to be published IN THE THESIS (in the appendix),
# because it is a methodological decision. Feel free to extend it based on
# your data — the frequency analysis in notebook 01 shows which words are
# missing.
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

# Pre-compiled regexes — with word boundaries, so that "bar" does not match
# inside "barely". For multi-word expressions the space matches too.
_ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    for aspect, kws in ASPECT_KEYWORDS.items()
}


# ==========================================================================
# 2) SENTENCE SPLITTING
# ==========================================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """
    Simple sentence splitting. For hotel reviews this is usually good
    enough, because the texts are short and simply structured.

    If more precision is needed (abbreviations, decimal points), use NLTK:
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
    Sentence-level data WITH KNOWN POLARITY — from the structure of the dataset.

    Why is this much better than projecting the review-level label?
    ---------------------------------------------------------------
    `explode_to_sentences()` puts the label derived from the REVIEW's overall
    score on every sentence. That is an approximation, and it fails exactly
    on the motivating example of your thesis: a 6.5-point ("neutral") review
    can contain both a clearly positive and a clearly negative sentence.

    The Booking data, however, stores the praise and the complaint in TWO
    SEPARATE FIELDS. If a sentence from the `review_positive` field is taken
    as positive and one from `review_negative` as negative, the
    sentence-level label is not an estimate — it follows from the structure
    of the data collection itself: the guest decided which field to write in.

    From ~200,000 reviews this gives several hundred thousand sentence-level,
    aspect-annotatable examples — orders of magnitude more than the 23,113
    sentences of HRAST.

    ⚠ LIMITATIONS that MUST be described in the thesis:
      1. This is the origin of the field, not human annotation. Sometimes
         someone writes mild criticism into the positive field ("great,
         although a bit small"). A noisy label, but not systematically biased.
      2. There is NO neutral class — the field structure is binary. For the
         3-class task this is not enough, the review-level labelling stays.
      3. HRAST is still needed: it contains REAL, human aspect annotation on
         which the model can be EVALUATED. This data is good for training and
         large-sample analysis, not as an evaluation gold standard.

    Returns
    -------
    One sentence per row: sentence_id, review_id, sentence, polarity
    ("positive" / "negative"), source_field, + the metadata columns.
    """
    missing = {pos_col, neg_col} - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing column(s): {sorted(missing)}. This function needs the "
            "separate positive/negative fields of Booking — check that the "
            "loader carried them over."
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

    log.info("Sentences by polarity: %s",
             out["polarity"].value_counts().to_dict())
    return out


def aspect_polarity_table(
    sent_df: pd.DataFrame,
    polarity_col: str = "polarity",
    min_count: int = 50,
) -> pd.DataFrame:
    """
    Per aspect: how often is it mentioned as praise, how often as a complaint?

    This is the CLEANEST answer to RQ3, because the polarity is not estimated
    but comes from the structure of the data (see `explode_by_polarity`).

    The `as_complaint` column: what percentage of the mentions of the aspect
    fall into the negative field. Above 0.5 it drives complaints, below it
    drives praise. `complaint_lift` compares this to the overall average.
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
            "aspect": aspect,
            "mentions": len(sub),
            "as_praise": 1 - neg,
            "as_complaint": neg,
            "complaint_lift": neg / base if base else float("nan"),
        })

    cols = ["aspect", "mentions", "as_praise", "as_complaint", "complaint_lift"]
    if not rows:
        log.warning("No aspect reached the min_count=%d threshold.", min_count)
        return pd.DataFrame(columns=cols).set_index("aspect")

    return (pd.DataFrame(rows)[cols]
            .set_index("aspect")
            .sort_values("complaint_lift", ascending=False)
            .round(4))


def explode_to_sentences(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Turn a review-level table into a sentence-level table.

    This is the first step of ABSA: aspect-sentiment pairs are interpreted at
    sentence level, because one review can contain several contradicting
    opinions (this is exactly the motivating example of the thesis!).
    """
    out = df.copy()
    out["sentences"] = out[text_col].map(split_sentences)
    out = out.explode("sentences").dropna(subset=["sentences"])
    out = out.rename(columns={"sentences": "sentence"}).reset_index(drop=True)
    out["sentence_id"] = out["review_id"].astype(str) + "_s" + out.groupby("review_id").cumcount().astype(str)
    return out


# ==========================================================================
# 3) ASPECT DETECTION
# ==========================================================================
def detect_aspects(text: str) -> list[str]:
    """Return the aspects whose keywords occur in the text."""
    if not isinstance(text, str) or not text:
        return []
    return [a for a, pat in _ASPECT_PATTERNS.items() if pat.search(text)]


def add_aspect_columns(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Add one 0/1 column per aspect (`aspect_room`, `aspect_staff`, ...),
    plus a list of all aspects found.
    """
    out = df.copy()
    out["aspects"] = out[text_col].map(detect_aspects)
    for aspect in CFG.aspects:
        out[f"aspect_{aspect}"] = out["aspects"].map(lambda lst, a=aspect: int(a in lst))
    out["n_aspects"] = out["aspects"].map(len)
    return out


# ==========================================================================
# 4) ASPECT × SENTIMENT ANALYSIS  (THE DIRECT ANSWER TO RQ3)
# ==========================================================================
def aspect_sentiment_table(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    min_count: int = 30,
) -> pd.DataFrame:
    """
    For each aspect: in how many sentences it occurs, and what the sentiment
    distribution is in those sentences.

    This table is one of the MOST IMPORTANT results of the thesis: it shows
    that e.g. 'noise' and 'cleanliness' are disproportionately often negative,
    while 'location' is overwhelmingly positive.

    The "negative excess" (neg_lift) column: how much more frequent negative
    sentiment is for this aspect than on average in the dataset.
    Above 1.0 = the aspect drives complaints; below 1.0 = it drives satisfaction.
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
            "aspect": aspect,
            "mentions": len(sub),
            "mention_share": len(sub) / len(df),
            "negative": neg,
            "neutral": neu,
            "positive": pos,
            "neg_lift": neg / base_neg if base_neg else float("nan"),
            "pos_lift": pos / base_pos if base_pos else float("nan"),
        })

    cols = ["aspect", "mentions", "mention_share",
            "negative", "neutral", "positive", "neg_lift", "pos_lift"]

    if not rows:
        # No aspect reached the min_count threshold. The notebook should not
        # stop — an empty table with the correct structure is returned.
        log.warning(
            "No aspect reached the min_count=%d threshold. "
            "Lower min_count or increase the sample.", min_count
        )
        return pd.DataFrame(columns=cols).set_index("aspect")

    return (pd.DataFrame(rows)[cols]
            .set_index("aspect")
            .sort_values("neg_lift", ascending=False)
            .round(4))


def plot_aspect_sentiment(table: pd.DataFrame, save_as: str | None = None):
    """Stacked bar chart: sentiment distribution per aspect."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    if table.empty:
        log.warning("The aspect table is empty, nothing to plot.")
        return None

    from .config import SENTIMENT_COLORS, INK

    data = table[["negative", "neutral", "positive"]].sort_values("negative", ascending=True)
    # The axis is CATEGORICAL: the index has to be converted to strings,
    # otherwise matplotlib positions the bars as numbers (with a numeric index
    # this makes the bars overlap).
    labels = [str(i) for i in data.index]

    fig, ax = plt.subplots(figsize=(8.5, max(3.6, 0.42 * len(data) + 2.2)))
    left = pd.Series(0.0, index=data.index)
    for col in ["negative", "neutral", "positive"]:
        ax.barh(labels, data[col], left=left, label=col,
                color=SENTIMENT_COLORS[col], height=0.72,
                edgecolor=INK["surface"], linewidth=1.5)
        left += data[col]

    ax.set_xlabel("Sentiment distribution in the sentences mentioning the aspect")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Sentiment distribution per aspect", loc="left", pad=34,
                 color=INK["primary"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 5) TEXTUAL EVIDENCE (RQ4 baseline)
# ==========================================================================
_NEGATORS = {"no", "not", "never", "without", "lack", "lacked", "lacking",
             "n't", "nothing", "none", "hardly", "barely"}


def extract_evidence(sentence: str, aspect: str, window: int = 4) -> list[str]:
    """
    Extract the word context around the aspect keyword — this is the simplest
    "textual evidence" (level 3 of chapter 10 of the thesis).

    Example:
        extract_evidence("the room was extremely hot because there was no air conditioning", "room")
        -> ["the room was extremely hot because"]

    The transformer-based version (notebook 04) will be much better, because
    it uses the attention weights or a token-classification head.
    This function is the BASELINE to compare against.
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
    """Is there a negation in the text snippet? The main weakness of keyword ABSA."""
    if not isinstance(text, str):
        return False
    toks = {t.lower().strip(".,!?;:") for t in text.split()}
    return bool(toks & _NEGATORS) or "n't" in text.lower()


# ==========================================================================
# 6) SENTENCE-PAIR FORMAT FOR TRANSFORMER ABSA
# ==========================================================================
def build_absa_pairs(
    df: pd.DataFrame,
    sentence_col: str = "sentence",
    aspect_label_col: str | None = None,
) -> pd.DataFrame:
    """
    Prepare the data for transformer ABSA (notebook 04).

    Output per row: (aspect, sentence, [label])
    In the notebook the tokenisation will look like this:

        tokenizer(batch["aspect"], batch["sentence"], truncation=True)

    With two arguments the HuggingFace tokenizer automatically produces the
    [CLS] aspect [SEP] sentence [SEP] format.
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
