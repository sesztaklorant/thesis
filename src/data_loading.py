"""
data_loading.py — Adatbetöltés és egységes séma / Data loading and a uniform schema.

HU: Két adatforrás:
    1. Booking.com val.csv — az ELEMZETT adat. Nincs benne szentiment-címke,
       csak 1–10-es vendégpontszám.
    2. HRAST — kézzel annotált, mondatszintű hotel-adat. KIZÁRÓLAG a (B)
       címkéző megközelítés tanítóadata.
EN: Two data sources:
    1. Booking.com val.csv — the ANALYSED data. It has no sentiment label,
       only a 1–10 guest score.
    2. HRAST — human-annotated, sentence-level hotel data. ONLY the training
       data of labelling approach (B).

HU: A Booking-betöltő egységes táblát ad vissza:
EN: The Booking loader returns a uniform table:
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

# HU: A cache-verzió a fájlnév része. Ha a betöltő logikája változik, ezt kell
#     növelni, és a régi cache automatikusan érvényét veszti.
# EN: The cache version is part of the file name. Increase it when the loader
#     logic changes, and the old cache is automatically ignored.
#   1 — HU: első változat / EN: initial version
#   2 — HU: HTML-entitások dekódolása, cím-írásjelek / EN: HTML entities decoded, title punctuation
#   3 — HU: review_title oszlop is mentve / EN: review_title column stored too
_CACHE_VERSION = 3


# ==========================================================================
# 0) PARQUET / CSV SEGÉDFÜGGVÉNYEK / PARQUET / CSV HELPERS
# ==========================================================================
def write_frame(df: pd.DataFrame, path: Path) -> Path:
    """
    HU: Parquet-mentés; ha a pyarrow hiányzik, CSV-re esik vissza (a pipeline nem áll le).
    EN: Save as parquet; falls back to CSV if pyarrow is missing (the pipeline does not stop).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception as e:                                        # noqa: BLE001
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False, encoding="utf-8")
        log.warning("Parquet not available (%s), saved as CSV: %s", e.__class__.__name__, csv.name)
        return csv


def read_frame(path: Path) -> pd.DataFrame:
    """HU: Beolvasás parquet- vagy CSV-formában. / EN: Read in parquet or CSV form."""
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
    return PATHS.raw / f"{name}__v{_CACHE_VERSION}.parquet"


def clear_cache(name: str | None = None) -> list[Path]:
    """
    HU: A betöltő által készített cache-fájlok törlése (az eredeti CSV-ket nem érinti).
    EN: Delete the cache files created by the loader (the original CSVs are untouched).
    """
    PATHS.ensure()
    pattern = f"{name}*__v*.parquet" if name else "*__v*.parquet"
    removed = []
    for p in PATHS.raw.glob(pattern):
        p.unlink()
        removed.append(p)
        log.info("Deleted: %s", p.name)
    return removed


# ==========================================================================
# 1) BOOKING.COM — HELYI CSV / LOCAL CSV
# ==========================================================================
# HU: A Booking CSV-k oszlopnevei exportonként eltérhetnek, ezért álnevekkel dolgozunk.
# EN: Column names differ between Booking exports, so we work with aliases.
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

# HU: Helykitöltő szövegek ("Nothing", "n/a"), amelyek valójában üres mezőt jelentenek.
# EN: Placeholder texts ("Nothing", "n/a") that actually mean an empty field.
_PLACEHOLDERS = {"nothing", "n/a", "na", "none", "-", "nan", "no negative",
                 "no positive", "nothing to report", "nothing really"}


def _normalise_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_columns(columns, aliases: dict[str, list[str]]) -> dict[str, str]:
    """
    HU: Megkeresi, melyik valós oszlop felel meg az egyes kanonikus mezőknek
        (kis-/nagybetűre és szóköz/aláhúzás különbségre érzéketlen).
    EN: Find which real column corresponds to each canonical field
        (insensitive to case and to the space/underscore difference).

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
    HU: Helykitöltő -> üres szöveg, HTML-entitások dekódolása (wouldn&#39;t -> wouldn't).
    EN: Placeholder -> empty string, decode HTML entities (wouldn&#39;t -> wouldn't).
    """
    if pd.isna(value):
        return ""
    value = html.unescape(str(value)).strip()
    return "" if value.lower() in _PLACEHOLDERS else value


def assemble_review_text(title: pd.Series, positive: pd.Series, negative: pd.Series) -> pd.Series:
    """
    HU: "cím. pozitív rész negatív rész" összeállítása; pont csak akkor kerül a cím
        után, ha az nem végződik már írásjelre.
    EN: Build "title. positive part negative part"; a full stop is added after the
        title only if it does not already end in punctuation.
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
    HU: A Booking.com CSV betöltése a data/raw/ mappából (alapból val.csv).
    EN: Load the Booking.com CSV from the data/raw/ folder (val.csv by default).

    Parameters
    ----------
    filename : HU: fájlnév a data/raw/ mappában / EN: file name in data/raw/
    n_rows   : HU: hány sort olvasson (None = mind) / EN: how many rows to read (None = all)
    """
    filename = filename or CFG.local_csv_name
    cache = _cache_path(f"booking_local_{Path(filename).stem}")

    if use_cache and n_rows is None and cache.exists():
        try:
            log.info("Loading from cache: %s", cache.name)
            return pd.read_parquet(cache)
        except Exception:                                         # noqa: BLE001
            log.warning("The cache cannot be read, re-reading the CSV.")

    PATHS.ensure()
    csv_path = PATHS.raw / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Not found: {csv_path}\nPut the CSV into data/raw/ or pass the exact "
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
        except Exception as e:                                    # noqa: BLE001
            log.warning("Caching failed (%s) — pip install pyarrow", e.__class__.__name__)
    return out


# ==========================================================================
# 2) HRAST — CSAK A (B) MEGKÖZELÍTÉSHEZ / ONLY FOR APPROACH (B)
# ==========================================================================
# HU: A HRAST pontos oszlopneveit letöltés után ELLENŐRIZNI kell. A betöltő
#     álnevekkel keres, és ha nem talál, kiírja a valós oszlopokat — ilyenkor
#     a prepare_hrast_sentences() paramétereivel kézzel megadhatók.
# EN: The exact HRAST column names have to be CHECKED after download. The
#     loader searches with aliases and, if it fails, prints the real columns —
#     they can then be given by hand via prepare_hrast_sentences().
_HRAST_ALIASES: dict[str, list[str]] = {
    "text":      ["sentence", "text", "review_sentence", "sentences", "content", "review"],
    "sentiment": ["sentiment", "polarity", "label", "sentiment_label",
                  "aspect_sentiment", "sentiment_polarity"],
    "aspect":    ["aspect", "aspect_category", "category", "aspects", "aspect_term"],
    "group":     ["review_id", "reviewid", "doc_id", "document_id", "hotel_review_id", "id_review"],
}

# HU: Címkeértékek normalizálása a kanonikus negative/neutral/positive formára.
# EN: Normalising label values to the canonical negative/neutral/positive form.
_SENTIMENT_VALUE_MAP: dict[str, str] = {
    "positive": "positive", "pos": "positive", "p": "positive", "+": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", "none": "neutral",
    "1": "positive", "-1": "negative", "1.0": "positive", "-1.0": "negative", "0.0": "neutral",
    "conflict": "conflict", "mixed": "conflict",
}


def find_hrast_file() -> Path | None:
    """
    HU: Keresés: data/raw/hrast.csv, data/raw/HRAST.csv vagy data/raw/hrast/*.csv.
    EN: Search: data/raw/hrast.csv, data/raw/HRAST.csv or data/raw/hrast/*.csv.
    """
    PATHS.ensure()
    for name in ("hrast.csv", "HRAST.csv"):
        if (PATHS.raw / name).exists():
            return PATHS.raw / name
    folder = PATHS.raw / "hrast"
    if folder.is_dir():
        csvs = sorted(folder.glob("*.csv"))
        if csvs:
            if len(csvs) > 1:
                log.warning("Several CSVs in data/raw/hrast/, using the first: %s",
                            [c.name for c in csvs])
            return csvs[0]
    return None


def load_hrast_raw(path: Path | str | None = None, try_kagglehub: bool = True) -> pd.DataFrame:
    """
    HU: A nyers HRAST-tábla betöltése, átalakítás nélkül. Kiírja az oszlopneveket.
    EN: Load the raw HRAST table without transformation. Prints the column names.
    """
    path = Path(path) if path else find_hrast_file()
    if path is None and try_kagglehub:
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(DATASETS["hrast"]["id"]))
            csvs = sorted(folder.rglob("*.csv"))
            path = csvs[0] if csvs else None
        except Exception as e:                                    # noqa: BLE001
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
    HU: Tetszőleges címkeértékek -> negative / neutral / positive / conflict (vagy NaN).
    EN: Arbitrary label values -> negative / neutral / positive / conflict (or NaN).
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
    HU: Egy mondat több aspektus-címkéjéből egy mondatszintű címke:
          csak pozitív (és semleges)  -> positive
          csak negatív (és semleges)  -> negative
          csak semleges               -> neutral
          pozitív ÉS negatív          -> conflict_policy szerint (drop / neutral)
    EN: One sentence-level label from the aspect labels of a sentence:
          only positive (and neutral) -> positive
          only negative (and neutral) -> negative
          only neutral                -> neutral
          positive AND negative       -> by conflict_policy (drop / neutral)
    """
    polar = set(labels) & {"positive", "negative"}
    if "conflict" in set(labels) or len(polar) == 2:
        return "neutral" if conflict_policy == "neutral" else np.nan
    if len(polar) == 1:
        return polar.pop()
    return "neutral"


def labels_from_onehot(raw: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    """
    HU: Három 0/1 oszlop -> egy címke soronként:
          pontosan egy 1-es          -> az az osztály
          semleges + egy polaritás   -> a polaritás (mint az aspektusoknál)
          pozitív ÉS negatív is 1    -> "conflict" (a conflict_policy dönt róla)
          egyik sem                  -> NaN (kimarad)
    EN: Three 0/1 columns -> one label per row:
          exactly one 1              -> that class
          neutral + one polarity     -> the polarity (as for the aspects)
          positive AND negative are 1 -> "conflict" (handled by conflict_policy)
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
    HU: A HRAST átalakítása mondatszintű, 3 osztályos tanítóadattá.
    EN: Turn HRAST into sentence-level, 3-class training data.

    Returns
    -------
    sentence | label | group_id   (HU: group_id = vélemény-azonosító, ha van —
                                   a felosztásnál egy vélemény mondatai együtt maradnak)
                                  (EN: group_id = review id if available — sentences
                                   of one review stay together in the split)
    """
    found = resolve_columns(raw.columns, _HRAST_ALIASES)
    text_col = text_col or found.get("text")
    sentiment_col = sentiment_col or found.get("sentiment")
    group_col = group_col or found.get("group")

    # HU: A Kaggle-féle HRAST "széles" formátumú: a szentiment három 0/1 oszlopban van
    #     (positive / negative / neutral), soronként egy mondat. Ezt is felismerjük.
    # EN: The Kaggle HRAST is in "wide" format: sentiment is in three 0/1 columns
    #     (positive / negative / neutral), one sentence per row. This is recognised too.
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

    # HU: Egy mondat több sorban is szerepelhet (soronként egy aspektus) -> összevonás.
    # EN: A sentence may appear in several rows (one aspect per row) -> merge.
    # HU: Ha van vélemény-azonosító, (vélemény, mondat) párra vonunk össze, így két
    #     különböző vélemény azonos mondata ("Great location.") nem olvad egybe.
    # EN: With a review id we merge on (review, sentence), so the same sentence in
    #     two different reviews ("Great location.") is not merged.
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
# 3) FELOSZTÁSOK ÉS CÍMKÉK MENTÉSE / SAVING SPLITS AND LABELS
# ==========================================================================
SPLIT_NAMES = ("train", "val", "test")


def save_splits(splits: dict[str, pd.DataFrame], prefix: str = "booking") -> dict[str, Path]:
    """HU: train/val/test mentése a data/processed/ mappába. / EN: Save train/val/test to data/processed/."""
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
