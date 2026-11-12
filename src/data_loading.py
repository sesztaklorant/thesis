"""
Loading of the two data sources into a common schema.

- Booking.com val.csv (data/raw/booking/): the analysed reviews, with a 1–10 guest
  score but no sentiment label.
- HRAST (data/raw/hrast/): hand-annotated hotel review sentences, used only to train
  labelling approach (B).

The Booking loader returns
review_id | text | score | score_scale_max | dataset | review_positive | review_negative | metadata...
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, DATASETS, PATHS

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["review_id", "text", "score", "score_scale_max", "dataset"]

# The cache version is part of the file name, so a change in the loader logic
# invalidates the old cache.
# 1 - initial version
# 2 - HTML entities decoded, title punctuation
# 3 - review_title column stored too
_CACHE_VERSION = 3


# ==========================================================================
# 0) PARQUET / CSV HELPERS
# ==========================================================================
def write_frame(df: pd.DataFrame, path: Path) -> Path:
    """
    Save as parquet, or as CSV if pyarrow is not installed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception as e:  # noqa: BLE001
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False, encoding="utf-8")
        log.warning("Parquet not available (%s), saved as CSV: %s", e.__class__.__name__, csv.name)
        return csv


def read_frame(path: Path) -> pd.DataFrame:
    """Read in parquet or CSV form."""
    path = Path(path)
    for candidate in (path.with_suffix(".parquet"), path.with_suffix(".csv")):
        if candidate.exists():
            return (pd.read_parquet(candidate) if candidate.suffix == ".parquet"
                    else pd.read_csv(candidate))
    raise FileNotFoundError(f"Neither {path.with_suffix('.parquet').name} nor "
                            f"{path.with_suffix('.csv').name} exists in {path.parent}")


def frame_exists(path: Path) -> bool:
    path = Path(path)
    return path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists()


def _cache_path(name: str) -> Path:
    PATHS.ensure()
    return PATHS.interim / f"{name}__v{_CACHE_VERSION}.parquet"


def clear_cache(name: str | None = None) -> list[Path]:
    """
    Delete the cache files written by the loader. The original CSVs are not touched.
    """
    PATHS.ensure()
    pattern = f"{name}*__v*.parquet" if name else "*__v*.parquet"
    removed = []
    for p in PATHS.interim.glob(pattern):
        p.unlink()
        removed.append(p)
        log.info("Deleted: %s", p.name)
    return removed


# ==========================================================================
# 1) BOOKING.COM CSV
# ==========================================================================
# Column names differ between Booking exports, hence the aliases.
_BOOKING_ALIASES: dict[str, list[str]] = {
    "review_title":              ["review_title", "title"],
    "review_positive":           ["review_positive", "positive_review", "positive", "pros"],
    "review_negative":           ["review_negative", "negative_review", "negative", "cons"],
    "guest_score":               ["guest_score", "review_score", "score", "reviewer_score", "rating"],
    "guest_type":                ["guest_type", "traveller_type", "traveler_type"],
    "guest_country":             ["guest_country", "reviewer_nationality", "nationality"],
    "room_nights":               ["room_nights", "nights", "total_number_of_nights"],
    "month":                     ["month", "review_month", "stay_month"],
    "accommodation_id":          ["accommodation_id", "hotel_id", "property_id"],
    "accommodation_type":        ["accommodation_type", "property_type", "hotel_type"],
    "accommodation_score":       ["accommodation_score", "hotel_score", "average_score"],
    "accommodation_country":     ["accommodation_country", "hotel_country", "country"],
    "accommodation_star_rating": ["accommodation_star_rating", "star_rating", "stars"],
    "location_is_beach":         ["location_is_beach", "is_beach"],
    "location_is_ski":           ["location_is_ski", "is_ski"],
    "location_is_city_center":   ["location_is_city_center", "is_city_center", "is_city_centre"],
    "review_helpful_votes":      ["review_helpful_votes", "helpful_votes"],
}

METADATA_COLUMNS = [
    "guest_type", "guest_country", "room_nights", "month", "accommodation_id",
    "accommodation_type", "accommodation_score", "accommodation_country",
    "accommodation_star_rating", "location_is_beach", "location_is_ski",
    "location_is_city_center", "review_helpful_votes",
]

# Placeholder texts ("Nothing", "n/a") that actually mean an empty field.
_PLACEHOLDERS = {"nothing", "n/a", "na", "none", "-", "nan", "no negative",
                 "no positive", "nothing to report", "nothing really"}


def _normalise_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_columns(columns, aliases: dict[str, list[str]]) -> dict[str, str]:
    """
    Map each canonical field to the matching column of the file
    (case-insensitive, spaces and underscores treated alike).

    Returns: {canonical_name: actual_column_name}
    """
    actual = {_normalise_name(c): c for c in columns}
    found: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in actual:
                found[canonical] = actual[name]
                break
    return found


def clean_placeholder(value) -> str:
    """
    Placeholder -> empty string; HTML entities decoded (wouldn&#39;t -> wouldn't).
    """
    if pd.isna(value):
        return ""
    value = html.unescape(str(value)).strip()
    return "" if value.lower() in _PLACEHOLDERS else value


def assemble_review_text(title: pd.Series, positive: pd.Series, negative: pd.Series) -> pd.Series:
    """
    Build "title. praise complaint". A full stop is added after the title only if
    it does not end in punctuation.
    """
    separator = title.map(lambda t: "" if (not t or t[-1] in ".!?…") else ".")
    text = title + separator + " " + positive + " " + negative
    return (text.str.replace(r"^[\.\s]+", "", regex=True)
                .str.replace(r"\.{2,}", ".", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())


def load_booking(
    filename: str | None = None,
    n_rows: int | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the Booking.com CSV from data/raw/booking/ (val.csv by default).

    Parameters
    ----------
    filename : file name in data/raw/booking/
    n_rows   : number of rows to read (None = all)
    """
    filename = filename or CFG.local_csv_name
    cache = _cache_path(f"booking_local_{Path(filename).stem}")

    if use_cache and n_rows is None and cache.exists():
        try:
            log.info("Loading from cache: %s", cache.name)
            return pd.read_parquet(cache)
        except Exception:  # noqa: BLE001
            log.warning("The cache cannot be read, re-reading the CSV.")

    PATHS.ensure()
    csv_path = PATHS.raw_booking / filename
    if not csv_path.exists() and (PATHS.raw / filename).exists():
        csv_path = PATHS.raw / filename  # earlier folder layout
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Not found: {csv_path}\nPut the CSV into data/raw/booking/ or pass the exact "
            f"file name: load_booking(filename='my_file.csv')"
        )

    log.info("Reading: %s", csv_path)
    raw = pd.read_csv(csv_path, nrows=n_rows, low_memory=False)

    cols = resolve_columns(raw.columns, _BOOKING_ALIASES)
    if verbose:
        log.info("Recognised columns: %d/%d", len(cols), len(_BOOKING_ALIASES))
        missing = sorted(set(_BOOKING_ALIASES) - set(cols))
        if missing:
            log.warning("Not found: %s | CSV columns: %s", missing, list(raw.columns))
    if "guest_score" not in cols:
        raise KeyError(f"No score column found. CSV columns: {list(raw.columns)}. "
                       "Extend _BOOKING_ALIASES in data_loading.py.")

    def _get(name: str) -> pd.Series:
        if name in cols:
            return raw[cols[name]].map(clean_placeholder)
        return pd.Series([""] * len(raw), index=raw.index)

    positive, negative, title = _get("review_positive"), _get("review_negative"), _get("review_title")

    out = pd.DataFrame({
        "text": assemble_review_text(title, positive, negative),
        "score": pd.to_numeric(raw[cols["guest_score"]], errors="coerce"),
        "review_title": title,
        "review_positive": positive,
        "review_negative": negative,
    })
    for meta in METADATA_COLUMNS:
        if meta in cols:
            out[meta] = raw[cols[meta]]

    out["dataset"] = CFG.source_split_name
    out["score_scale_max"] = float(DATASETS["booking"]["score_scale"][1])
    out["review_id"] = [f"{CFG.source_split_name}_{i}" for i in range(len(out))]
    out["text"] = out["text"].astype("string")
    extra = [c for c in out.columns if c not in CANONICAL_COLUMNS]
    out = out[CANONICAL_COLUMNS + extra]

    if n_rows is None:
        try:
            out.to_parquet(cache, index=False)
            log.info("Saved to cache: %s (%d rows)", cache.name, len(out))
        except Exception as e:  # noqa: BLE001
            log.warning("Caching failed (%s) — pip install pyarrow", e.__class__.__name__)
    return out


# ==========================================================================
# 2) HRAST (TRAINING DATA OF APPROACH B)
# ==========================================================================
# Column names are matched with aliases. If no match is found, the actual columns are
# printed and can be passed to prepare_hrast_sentences().
_HRAST_ALIASES: dict[str, list[str]] = {
    "text":      ["sentence", "text", "review_sentence", "sentences", "content", "review"],
    "sentiment": ["sentiment", "polarity", "label", "sentiment_label",
                  "aspect_sentiment", "sentiment_polarity"],
    "aspect":    ["aspect", "aspect_category", "category", "aspects", "aspect_term"],
    "group":     ["review_id", "reviewid", "doc_id", "document_id", "hotel_review_id", "id_review"],
}

# Label values mapped to negative / neutral / positive.
_SENTIMENT_VALUE_MAP: dict[str, str] = {
    "positive": "positive", "pos": "positive", "p": "positive", "+": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", "none": "neutral",
    "1": "positive", "-1": "negative", "1.0": "positive", "-1.0": "negative", "0.0": "neutral",
    "conflict": "conflict", "mixed": "conflict",
}


def find_hrast_file() -> Path | None:
    """
    Locate the HRAST csv: data/raw/hrast/*.csv, or data/raw/hrast.csv / HRAST.csv.
    """
    PATHS.ensure()
    folder = PATHS.raw_hrast
    if folder.is_dir():
        csvs = sorted(folder.glob("*.csv"))
        if csvs:
            if len(csvs) > 1:
                log.warning("Several CSVs in data/raw/hrast/, using the first: %s",
                            [c.name for c in csvs])
            return csvs[0]
    for name in ("hrast.csv", "HRAST.csv"):  # earlier folder layout
        if (PATHS.raw / name).exists():
            return PATHS.raw / name
    return None


def load_hrast_raw(path: Path | str | None = None, try_kagglehub: bool = True) -> pd.DataFrame:
    """
    Load the raw HRAST table without changes and log its column names.
    """
    path = Path(path) if path else find_hrast_file()
    if path is None and try_kagglehub:
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(DATASETS["hrast"]["id"]))
            csvs = sorted(folder.rglob("*.csv"))
            path = csvs[0] if csvs else None
        except Exception as e:  # noqa: BLE001
            log.warning("kagglehub download failed (%s).", e)
    if path is None:
        raise FileNotFoundError(
            "HRAST not found. Download it from Kaggle "
            f"({DATASETS['hrast']['id']}) and put the CSV into data/raw/hrast/."
        )
    raw = pd.read_csv(path)
    log.info("HRAST loaded: %s — %d rows, columns: %s", Path(path).name, len(raw), list(raw.columns))
    return raw


def normalise_sentiment_values(values: pd.Series, label_map: dict | None = None) -> pd.Series:
    """
    Label values of any form -> negative / neutral / positive / conflict (or NaN).
    """
    mapping = {**_SENTIMENT_VALUE_MAP, **{str(k).lower(): v for k, v in (label_map or {}).items()}}
    normalised = values.astype(str).str.strip().str.lower().map(mapping)
    unknown = sorted(values[normalised.isna() & values.notna()].astype(str).unique())[:10]
    if unknown:
        log.warning("Unknown sentiment values (dropped): %s — pass label_map={...} to map them.",
                    unknown)
    return normalised


def _sentence_label_from_aspects(labels: pd.Series, conflict_policy: str) -> str | float:
    """
    Sentence label derived from the aspect labels of the sentence:
      only positive (and neutral) -> positive
      only negative (and neutral) -> negative
      only neutral                -> neutral
      positive and negative       -> by conflict_policy (drop / neutral)
    """
    polar = set(labels) & {"positive", "negative"}
    if "conflict" in set(labels) or len(polar) == 2:
        return "neutral" if conflict_policy == "neutral" else np.nan
    if len(polar) == 1:
        return polar.pop()
    return "neutral"


def labels_from_onehot(raw: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    """
    Three 0/1 columns -> one label per row:
      exactly one 1              -> that class
      neutral + one polarity     -> the polarity (as for the aspects)
      positive and negative are 1 -> "conflict" (handled by conflict_policy)
      none                       -> NaN (dropped)
    """
    flags = pd.DataFrame({label: pd.to_numeric(raw[col], errors="coerce").fillna(0) > 0
                          for label, col in columns.items()})
    pos, neg, neu = flags["positive"], flags["negative"], flags["neutral"]
    label = pd.Series(np.nan, index=raw.index, dtype=object)
    label[neu & ~pos & ~neg] = "neutral"
    label[pos & ~neg] = "positive"
    label[neg & ~pos] = "negative"
    label[pos & neg] = "conflict"
    log.info("HRAST one-hot rows — positive: %d | negative: %d | neutral only: %d | "
             "both pos+neg: %d | none: %d", int((pos & ~neg).sum()), int((neg & ~pos).sum()),
             int((neu & ~pos & ~neg).sum()), int((pos & neg).sum()), int((~pos & ~neg & ~neu).sum()))
    return label


def prepare_hrast_sentences(
    raw: pd.DataFrame,
    text_col: str | None = None,
    sentiment_col: str | None = None,
    group_col: str | None = None,
    label_map: dict | None = None,
    conflict_policy: str = CFG.hrast_conflict_policy,
) -> pd.DataFrame:
    """
    Convert HRAST to sentence-level, 3-class training data.

    Returns
    -------
    sentence | label | group_id   (group_id = review id if available, so the
                                   sentences of a review stay in the same split)
    """
    found = resolve_columns(raw.columns, _HRAST_ALIASES)
    text_col = text_col or found.get("text")
    sentiment_col = sentiment_col or found.get("sentiment")
    group_col = group_col or found.get("group")

    # The Kaggle version of HRAST is in wide format: three 0/1 sentiment columns
    # (positive / negative / neutral) and one sentence per row.
    onehot = resolve_columns(raw.columns, {label: [label] for label in CFG.labels})
    use_onehot = sentiment_col is None and len(onehot) == len(CFG.labels)

    if text_col is None or (sentiment_col is None and not use_onehot):
        raise KeyError(
            f"Text / sentiment column not recognised. HRAST columns: {list(raw.columns)}\n"
            "Pass them explicitly: prepare_hrast_sentences(raw, text_col='...', sentiment_col='...')"
        )
    log.info("HRAST columns used — text: %s | sentiment: %s | group: %s", text_col,
             f"one-hot {list(onehot.values())}" if use_onehot else sentiment_col, group_col)

    label_raw = (labels_from_onehot(raw, onehot) if use_onehot
                 else normalise_sentiment_values(raw[sentiment_col], label_map))
    work = pd.DataFrame({
        "sentence": raw[text_col].astype(str).map(html.unescape).str.strip(),
        "label_raw": label_raw,
        "group_id": raw[group_col].astype(str) if group_col else np.nan,
    }).dropna(subset=["label_raw"])
    work = work[work["sentence"].str.len() > 0]

    # A sentence can appear in several rows (one aspect per row), so rows are merged.
    # With a review id the key is (review, sentence), which keeps identical sentences
    # of different reviews ("Great location.") apart.
    keys = ["group_id", "sentence"] if group_col else ["sentence"]
    grouped = (work.groupby(keys, sort=False, dropna=False)
                   .agg(labels=("label_raw", list))
                   .reset_index())
    if "group_id" not in grouped.columns:
        grouped["group_id"] = np.nan
    grouped["label"] = grouped["labels"].map(
        lambda lbls: _sentence_label_from_aspects(pd.Series(lbls), conflict_policy))

    n_before = len(grouped)
    out = grouped.dropna(subset=["label"])[["sentence", "label", "group_id"]].reset_index(drop=True)
    log.info("HRAST sentences: %d (dropped as conflicting: %d) | distribution: %s",
             len(out), n_before - len(out), out["label"].value_counts().to_dict())
    return out


# ==========================================================================
# 3) SAVING SPLITS AND LABELS
# ==========================================================================
SPLIT_NAMES = ("train", "val", "test")


def save_splits(splits: dict[str, pd.DataFrame], prefix: str = "booking") -> dict[str, Path]:
    """Save train/val/test to data/processed/."""
    PATHS.ensure()
    written = {}
    for name, part in splits.items():
        written[name] = write_frame(part, PATHS.processed / f"{prefix}_{name}")
        log.info("%-6s -> %-30s (%d rows)", name, written[name].name, len(part))
    return written


def load_split(name: str, prefix: str = "booking") -> pd.DataFrame:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {name}. Choose from: {SPLIT_NAMES}")
    return read_frame(PATHS.processed / f"{prefix}_{name}")


def load_all_splits(prefix: str = "booking") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(load_split(n, prefix) for n in SPLIT_NAMES)  # type: ignore[return-value]
