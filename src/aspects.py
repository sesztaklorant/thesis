"""Keyword-based aspect detection and aspect-level sentiment.

A review can contain opposite opinions ("the room is great, the breakfast is bad"),
so aspects are assigned to sentences. The sentence sentiment comes either from the
sentence-level BERT prediction or from the Booking field the sentence was written in.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .config import CFG, INK, SENTIMENT_COLORS
from .preprocessing import split_sentences

log = logging.getLogger(__name__)


# ==========================================================================
# 1) ASPECT LEXICON
# ==========================================================================
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "room": ["room", "rooms", "bedroom", "suite", "bed", "beds", "mattress", "pillow", "pillows",
             "sheets", "linen", "air conditioning", "aircon", "a/c", "heating", "radiator",
             "window", "balcony", "view"],
    "staff": ["staff", "reception", "receptionist", "employee", "employees", "manager", "service",
              "housekeeping", "concierge", "waiter", "waitress", "front desk", "host", "team",
              "personnel"],
    "cleanliness": ["clean", "cleanliness", "dirty", "dust", "dusty", "stain", "stains", "smell",
                    "smelly", "odour", "odor", "mold", "mould", "hygiene", "hygienic", "filthy",
                    "spotless", "tidy", "cockroach", "bed bugs"],
    "facilities": ["facility", "facilities", "gym", "fitness", "spa", "sauna", "elevator", "lift",
                   "restaurant", "bar", "lobby", "amenities", "kettle", "fridge", "minibar", "tv",
                   "television"],
    "location": ["location", "located", "situated", "central", "centre", "center", "downtown",
                 "station", "metro", "bus", "airport", "walking distance", "close to", "near",
                 "nearby", "neighbourhood", "neighborhood", "transport", "beach", "city center",
                 "city centre"],
    "breakfast": ["breakfast", "buffet", "coffee", "croissant", "food", "meal", "dinner", "lunch"],
    "pool": ["pool", "swimming", "jacuzzi", "hot tub"],
    "bathroom": ["bathroom", "shower", "toilet", "bath", "sink", "towel", "towels", "hot water",
                 "water pressure", "toiletries", "hairdryer"],
    "parking": ["parking", "car park", "garage", "valet"],
    "noise": ["noise", "noisy", "loud", "quiet", "soundproof", "soundproofing", "thin walls",
              "street noise", "traffic noise"],
    "wifi": ["wifi", "wi-fi", "wireless", "internet", "connection", "signal"],
    "value_for_money": ["price", "prices", "expensive", "cheap", "value", "money", "overpriced",
                        "worth", "cost", "affordable", "rate", "charge", "extra charge", "hidden fee"],
}

# With word boundaries, so that "bar" does not match inside "barely".
_ASPECT_PATTERNS = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")\b",
                       flags=re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}


def detect_aspects(text: str) -> list[str]:
    """The aspects occurring in the text."""
    if not isinstance(text, str) or not text:
        return []
    return [a for a, pattern in _ASPECT_PATTERNS.items() if pattern.search(text)]


def add_aspect_columns(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """One 0/1 column per aspect (aspect_room, ...)."""
    out = df.copy()
    out["aspects"] = out[text_col].map(detect_aspects)
    for aspect in CFG.aspects:
        out[f"aspect_{aspect}"] = out["aspects"].map(lambda found, a=aspect: int(a in found))
    out["n_aspects"] = out["aspects"].map(len)
    return out


# ==========================================================================
# 2) SENTENCES FROM THE BOOKING FIELDS
# ==========================================================================
def explode_by_polarity(df: pd.DataFrame, pos_col: str = "review_positive",
                        neg_col: str = "review_negative", min_words: int = 3) -> pd.DataFrame:
    """
    Sentences from the review_positive / review_negative fields with the
    field's polarity. This is not a human annotation but the guest's own choice (which
    field to write in) - a good independent reference for the BERT sentence
    labels. Limitation: there is no neutral class.
    """
    meta = [c for c in ["review_id", "score", "label_final", "accommodation_type",
                        "accommodation_country", "guest_type"] if c in df.columns]
    frames = []
    for col, polarity in [(pos_col, "positive"), (neg_col, "negative")]:
        part = df[meta + [col]].copy()
        part["sentence"] = part[col].map(lambda t: split_sentences(t, min_words) if isinstance(t, str) and t else [])
        part = part.drop(columns=[col]).explode("sentence").dropna(subset=["sentence"])
        part["field_polarity"] = polarity
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["sentence"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    log.info("Sentences by field polarity: %s", out["field_polarity"].value_counts().to_dict())
    return out


# ==========================================================================
# 3) ASPEKTUS × SZENTIMENT / ASPECT × SENTIMENT
# ==========================================================================
def aspect_sentiment_table(sent_df: pd.DataFrame, label_col: str, min_count: int = 30,
                           aspects: list[str] | None = None, prefix: str = "aspect_") -> pd.DataFrame:
    """
    Per aspect: how many sentences mention it and the sentiment distribution in them.
    neg_lift > 1: the aspect drives complaints; < 1: drives satisfaction.

    aspects / prefix: for the learned model's columns (aspect_model), e.g. prefix="asp_".
    """
    labels = sent_df[label_col].astype(str)
    base_neg, base_pos = (labels == "negative").mean(), (labels == "positive").mean()
    rows = []
    for aspect in (aspects or CFG.aspects):
        col = f"{prefix}{aspect}"
        if col not in sent_df.columns:
            continue
        sub = labels[sent_df[col] == 1]
        if len(sub) < min_count:
            continue
        dist = sub.value_counts(normalize=True)
        rows.append({"aspect": aspect, "mentions": len(sub), "mention_share": len(sub) / len(sent_df),
                     **{lab: dist.get(lab, 0.0) for lab in CFG.labels},
                     "neg_lift": dist.get("negative", 0.0) / base_neg if base_neg else float("nan"),
                     "pos_lift": dist.get("positive", 0.0) / base_pos if base_pos else float("nan")})
    cols = ["aspect", "mentions", "mention_share", *CFG.labels, "neg_lift", "pos_lift"]
    if not rows:
        log.warning("No aspect reached min_count=%d.", min_count)
        return pd.DataFrame(columns=cols).set_index("aspect")
    return pd.DataFrame(rows)[cols].set_index("aspect").sort_values("neg_lift", ascending=False).round(4)


def plot_aspect_sentiment(table: pd.DataFrame, title: str = "Sentiment distribution per aspect",
                          save_as: str | None = None):
    """Stacked bar chart per aspect."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    if table.empty:
        return None
    data = table[list(CFG.labels)].sort_values("negative")
    names = [str(i) for i in data.index]
    fig, ax = plt.subplots(figsize=(8.5, max(3.6, 0.42 * len(data) + 2.2)))
    left = pd.Series(0.0, index=data.index)
    for label in CFG.labels:
        ax.barh(names, data[label], left=left, label=label, color=SENTIMENT_COLORS[label],
                height=0.72, edgecolor=INK["surface"], linewidth=1.5)
        left += data[label]
    ax.set_xlabel("Sentiment distribution in the sentences mentioning the aspect")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 4) TEXTUAL EVIDENCE
# ==========================================================================
_NEGATORS = {"no", "not", "never", "without", "lack", "lacked", "lacking",
             "n't", "nothing", "none", "hardly", "barely"}


def extract_evidence(sentence: str, aspect: str, window: int = 4) -> list[str]:
    """
    The ±window word context around the keyword (simple "textual evidence").
    """
    if aspect not in ASPECT_KEYWORDS or not isinstance(sentence, str):
        return []
    tokens = sentence.split()
    lowered = [t.lower().strip(".,!?;:") for t in tokens]
    keywords = {k.lower() for k in ASPECT_KEYWORDS[aspect] if " " not in k}
    return [" ".join(tokens[max(0, i - window):i + window + 1])
            for i, tok in enumerate(lowered) if tok in keywords]


def has_negation(text: str) -> bool:
    """True if the snippet contains a negation."""
    if not isinstance(text, str):
        return False
    tokens = {t.lower().strip(".,!?;:") for t in text.split()}
    return bool(tokens & _NEGATORS) or "n't" in text.lower()
