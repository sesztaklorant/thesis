"""
data_loading.py — Downloading the data and bringing it to a uniform shape.

The datasets have very different structures. The goal is to produce the
SAME table from each of them, so that the downstream code (cleaning,
TF-IDF, BERT) does not need to know where the data came from:

    review_id | text | score | score_scale_max | dataset | (optional extra columns)

This makes the RQ5 (cross-dataset generalisation) experiment trivial: you
call the same function with a different `dataset` parameter.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import pandas as pd

from .config import CFG, PATHS, DATASETS

log = logging.getLogger(__name__)

# The uniform schema returned by every loader.
CANONICAL_COLUMNS = ["review_id", "text", "score", "score_scale_max", "dataset"]


# ==========================================================================
# Helper functions
# ==========================================================================
# --------------------------------------------------------------------------
# CACHE VERSION
# --------------------------------------------------------------------------
# The loaded data is saved to a parquet file, so the CSV does not have to be
# re-read the second time. This is fast, but it hides a trap: if the logic
# of the loader or of the text assembly changes, the old cache becomes stale
# and silently returns the OLD data — while the code is already new.
#
# Solution: the version number is part of the file name. When the number is
# increased here, the old files simply do not match and the loader re-reads
# the CSV. The old files stay on disk; clear_cache() removes them.
#
# Version history:
#   1 — initial version
#   2 — HTML entities decoded (&#39; -> '), punctuation handling of the title
_CACHE_VERSION = 2


def _cache_path(name: str) -> Path:
    """Where the loaded, still raw data is saved."""
    PATHS.ensure()
    return PATHS.raw / f"{name}__v{_CACHE_VERSION}.parquet"


def clear_cache(name: str | None = None) -> list[Path]:
    """
    Delete the cache files from the data/raw/ folder.

    Does not touch the ORIGINAL downloaded CSVs — only the `*__v*.parquet`
    files created by this module.

        dl.clear_cache()                 # all of them
        dl.clear_cache("booking_local")  # only this one
    """
    PATHS.ensure()
    pattern = f"{name}*__v*.parquet" if name else "*__v*.parquet"
    removed = []
    for p in PATHS.raw.glob(pattern):
        p.unlink()
        removed.append(p)
        log.info("Deleted: %s", p.name)
    if not removed:
        log.info("No cache file found to delete.")
    return removed


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """
    Write the cache in parquet format.

    Parquet is fast and small, but it needs the pyarrow package. If that is
    not available for some reason, caching is simply skipped — the code keeps
    working, every load just re-reads the CSV.
    """
    try:
        df.to_parquet(path, index=False)
        log.info("Saved to cache: %s (%d rows)", path.name, len(df))
    except Exception as e:                                        # noqa: BLE001
        log.warning("Caching failed (%s). The code keeps working, reloading "
                    "will just be slower. Fix: pip install pyarrow",
                    e.__class__.__name__)


def _finalize(df: pd.DataFrame, dataset: str, scale_max: float) -> pd.DataFrame:
    """Uniform column names + types + sequential identifier."""
    df = df.copy()
    df["dataset"] = dataset
    df["score_scale_max"] = float(scale_max)
    df["review_id"] = [f"{dataset}_{i}" for i in range(len(df))]
    df["text"] = df["text"].astype("string")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    return df[CANONICAL_COLUMNS + extra]


# ==========================================================================
# 1) Booking.com — Hugging Face (MAIN dataset)
# ==========================================================================
def load_booking_hf(
    n_rows: int | None = None,
    use_cache: bool = True,
    combine_pos_neg: bool = True,
) -> pd.DataFrame:
    """
    Booking.com Accommodation Reviews (CC BY-SA 4.0).

    Parameters
    ----------
    n_rows : how many rows to load (None = all ~1.6M). For development give
             e.g. 50_000, otherwise it takes long and eats a lot of memory.
    combine_pos_neg : if True, the `review_positive` and `review_negative`
             fields are joined into a single text. This corresponds to the
             "review-level" classification described in the thesis.

             NOTE (worth mentioning in the thesis as well): each of the two
             fields already carries a sentiment label on its own. If you use
             them together AND derive the label from the score, that is
             consistent. But if you split them into separate sentences, the
             sentences from the "positive" field are positive almost by
             definition — this is not a bug, it just has to be documented.
    """
    meta = DATASETS["booking_hf"]
    cache = _cache_path("booking_hf")

    if use_cache and cache.exists():
        log.info("Loading from cache: %s", cache)
        df = pd.read_parquet(cache)
        return df.head(n_rows) if n_rows else df

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is missing. Install it: pip install datasets"
        ) from e

    log.info("Downloading from Hugging Face: %s", meta["id"])
    # streaming=True: does not load all 1.6M rows into memory, yields row by row.
    # During development this is your friend.
    if n_rows:
        stream = load_dataset(meta["id"], split="train", streaming=True)
        rows = []
        for i, row in enumerate(stream):
            if i >= n_rows:
                break
            rows.append(row)
        raw = pd.DataFrame(rows)
    else:
        ds = load_dataset(meta["id"], split="train")
        raw = ds.to_pandas()

    # --- Assemble the text ---
    def _clean_part(s):
        """In the Booking data a missing part is often 'Nothing' / an empty string."""
        s = "" if pd.isna(s) else str(s).strip()
        return "" if s.lower() in {"nothing", "n/a", "na", "none", "-"} else s

    pos = raw.get("review_positive", pd.Series([""] * len(raw))).map(_clean_part)
    neg = raw.get("review_negative", pd.Series([""] * len(raw))).map(_clean_part)
    title = raw.get("review_title", pd.Series([""] * len(raw))).map(_clean_part)

    if combine_pos_neg:
        text = (title + ". " + pos + " " + neg).str.strip().str.replace(r"\s+", " ", regex=True)
    else:
        text = pos + " " + neg

    df = pd.DataFrame({
        "text": text,
        "score": raw[meta["score_field"]],
        # Useful extra fields for EDA and error analysis:
        "accommodation_type": raw.get("accommodation_type"),
        "accommodation_country": raw.get("accommodation_country"),
        "guest_type": raw.get("guest_type"),
        "review_positive": pos,
        "review_negative": neg,
    })

    df = _finalize(df, "booking_hf", meta["score_scale"][1])
    df.to_parquet(cache, index=False)
    log.info("Saved: %s (%d rows)", cache, len(df))
    return df


# ==========================================================================
# 1/b) Booking.com — LOCAL CSV (from the official train/val/test split)
# ==========================================================================
# The Booking.com dataset can also be downloaded split into three files:
# train.csv, val.csv, test.csv. As agreed with the supervisor, we start with
# the ~200k-row val.csv.
#
# ⚠ NAME CLASH — this has to be clarified in the thesis as well:
#   "val.csv" is Booking's OWN validation split. For us this file is the
#   WHOLE working dataset, which WE split 60/20/20 into our own
#   train/validation/test sets. The two have nothing to do with each other.
#   That is why the loaded dataset is called "booking_val_split", not
#   "validation".

# Possible column names of the Booking CSVs. Several variants exist
# (HF export, GitHub release), so we work with aliases.
_BOOKING_ALIASES: dict[str, list[str]] = {
    "review_title":             ["review_title", "title"],
    "review_positive":          ["review_positive", "positive_review", "positive", "pros"],
    "review_negative":          ["review_negative", "negative_review", "negative", "cons"],
    "guest_score":              ["guest_score", "review_score", "score",
                                 "reviewer_score", "rating"],
    "guest_type":               ["guest_type", "traveller_type", "traveler_type"],
    "guest_country":            ["guest_country", "reviewer_nationality", "nationality"],
    "room_nights":              ["room_nights", "nights", "total_number_of_nights"],
    "month":                    ["month", "review_month", "stay_month"],
    "accommodation_id":         ["accommodation_id", "hotel_id", "property_id"],
    "accommodation_type":       ["accommodation_type", "property_type", "hotel_type"],
    "accommodation_score":      ["accommodation_score", "hotel_score", "average_score"],
    "accommodation_country":    ["accommodation_country", "hotel_country", "country"],
    "accommodation_star_rating": ["accommodation_star_rating", "star_rating", "stars"],
    "location_is_beach":        ["location_is_beach", "is_beach"],
    "location_is_ski":          ["location_is_ski", "is_ski"],
    "location_is_city_center":  ["location_is_city_center", "is_city_center", "is_city_centre"],
    "review_helpful_votes":     ["review_helpful_votes", "helpful_votes"],
}

# These fields may contain placeholders like "Nothing"/"n/a".
_PLACEHOLDERS = {"nothing", "n/a", "na", "none", "-", "nan", "no negative",
                 "no positive", "nothing to report", "nothing really"}


def _resolve_columns(df_columns) -> dict[str, str]:
    """
    Find which actual column of the CSV corresponds to each canonical field.
    Insensitive to case and to the space/underscore difference.

    Returns: {canonical_name: actual_column_name}
    """
    def norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")

    actual = {norm(c): c for c in df_columns}
    found: dict[str, str] = {}
    for canonical, aliases in _BOOKING_ALIASES.items():
        for a in aliases:
            if a in actual:
                found[canonical] = actual[a]
                break
    return found


def _clean_placeholder(s) -> str:
    """
    Placeholder text -> empty string, and decode HTML entities.

    In the Booking export the apostrophe is HTML-encoded: `wouldn&#39;t`.
    It is decoded already here so that the raw `text` column is readable
    too — this way the examples printed in the EDA and the quotes that go
    into the thesis are fine as well, not only the cleaned versions.
    """
    if pd.isna(s):
        return ""
    s = html.unescape(str(s)).strip()
    return "" if s.lower() in _PLACEHOLDERS else s


def load_booking_local(
    filename: str | None = None,
    n_rows: int | None = None,
    combine_pos_neg: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the Booking.com CSV from the data/raw/ folder (val.csv by default).

    Parameters
    ----------
    filename : name of the file in data/raw/. If None, CFG.local_csv_name.
    n_rows   : how many rows to read (None = all). Useful for development.
    combine_pos_neg : whether to join the positive and negative text parts
                      into one field.

    Returns
    -------
    Canonical table: review_id, text, score, score_scale_max, dataset,
    + the metadata columns contained in the CSV (country, accommodation type,
    guest type, star rating, month...). The crosstabs are built on these.
    """
    filename = filename or CFG.local_csv_name
    cache = _cache_path(f"booking_local_{Path(filename).stem}")

    if use_cache and cache.exists() and n_rows is None:
        try:
            log.info("Loading from cache: %s", cache.name)
            return pd.read_parquet(cache)
        except Exception:                                         # noqa: BLE001
            log.warning("The cache cannot be read, re-reading the CSV.")

    PATHS.ensure()
    csv_path = PATHS.raw / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Not found: {csv_path}\n"
            f"Put the CSV into the data/raw/ folder, or give the exact file name:\n"
            f"    load_booking_local(filename='my_file.csv')"
        )

    log.info("Reading: %s", csv_path)
    raw = pd.read_csv(csv_path, nrows=n_rows, low_memory=False)

    # --- Recognise the columns ---
    cols = _resolve_columns(raw.columns)
    if verbose:
        log.info("Recognised columns (%d/%d):", len(cols), len(_BOOKING_ALIASES))
        for k, v in cols.items():
            log.info("    %-28s <- %s", k, v)
        missing = set(_BOOKING_ALIASES) - set(cols)
        if missing:
            log.warning("Not found: %s", sorted(missing))
            log.warning("Actual columns of the CSV: %s", list(raw.columns))

    if "guest_score" not in cols:
        raise KeyError(
            "The score column was not found. Columns of the CSV: "
            f"{list(raw.columns)}\nExtend the _BOOKING_ALIASES dictionary in "
            "data_loading.py with the right name."
        )

    # --- Assemble the text ---
    def _get(name: str) -> pd.Series:
        if name in cols:
            return raw[cols[name]].map(_clean_placeholder)
        return pd.Series([""] * len(raw), index=raw.index)

    pos, neg, title = _get("review_positive"), _get("review_negative"), _get("review_title")

    if combine_pos_neg:
        # A full stop is added after the title only if it does not already end
        # in punctuation — otherwise double stops like "Great stay! ... venue.."
        # appear, which look ugly in the quotes lifted into the thesis.
        sep = title.map(lambda t: "" if (not t or t[-1] in ".!?…") else ".")
        text = (title + sep + " " + pos + " " + neg)
    else:
        text = (pos + " " + neg)

    text = (text.str.replace(r"^[\.\s]+", "", regex=True)     # leading stop from an empty title
                .str.replace(r"\.{2,}", ".", regex=True)      # remaining double stops
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())

    out = pd.DataFrame({
        "text": text,
        "score": pd.to_numeric(raw[cols["guest_score"]], errors="coerce"),
        "review_positive": pos,
        "review_negative": neg,
    })

    # --- Metadata: the crosstabs are built on these ---
    for meta in ["guest_type", "guest_country", "room_nights", "month",
                 "accommodation_id", "accommodation_type", "accommodation_score",
                 "accommodation_country", "accommodation_star_rating",
                 "location_is_beach", "location_is_ski", "location_is_city_center",
                 "review_helpful_votes"]:
        if meta in cols:
            out[meta] = raw[cols[meta]]

    df = _finalize(out, CFG.source_split_name, DATASETS["booking_hf"]["score_scale"][1])

    if n_rows is None:
        _write_cache(df, cache)

    return df


# ==========================================================================
# 2) Kaggle 515K Hotel Reviews (external validation)
# ==========================================================================
def load_kaggle_515k(n_rows: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    515K Hotel Reviews Data in Europe.

    Download: automatically with kagglehub, OR manually from Kaggle
    (Hotel_Reviews.csv) into the data/raw/ folder.
    """
    meta = DATASETS["kaggle_515k"]
    cache = _cache_path("kaggle_515k")

    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        return df.head(n_rows) if n_rows else df

    csv = _find_local_csv(["Hotel_Reviews.csv"])
    if csv is None:
        csv = _kagglehub_download(meta["id"], want="Hotel_Reviews.csv")

    raw = pd.read_csv(csv, nrows=n_rows)

    def _clean_part(s, placeholder):
        s = "" if pd.isna(s) else str(s).strip()
        return "" if s.lower() == placeholder else s

    pos = raw["Positive_Review"].map(lambda s: _clean_part(s, "no positive"))
    neg = raw["Negative_Review"].map(lambda s: _clean_part(s, "no negative"))

    df = pd.DataFrame({
        "text": (pos + " " + neg).str.strip().str.replace(r"\s+", " ", regex=True),
        "score": raw["Reviewer_Score"],
        "hotel_name": raw.get("Hotel_Name"),
        "review_positive": pos,
        "review_negative": neg,
    })

    df = _finalize(df, "kaggle_515k", meta["score_scale"][1])
    df.to_parquet(cache, index=False)
    return df


# ==========================================================================
# 3) TripAdvisor (ALTERNATIVE external validation — DIFFERENT platform)
# ==========================================================================
def load_tripadvisor(n_rows: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    TripAdvisor Hotel Reviews (~20k, 1–5 stars).

    Why does it matter? The Booking.com and the 515K datasets come from the
    SAME platform, just from different periods. If you want to answer the
    "cross-dataset generalization" question (RQ5) seriously, data from
    another platform gives a much stronger test.
    """
    meta = DATASETS["tripadvisor"]
    cache = _cache_path("tripadvisor")

    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        return df.head(n_rows) if n_rows else df

    csv = _find_local_csv(["tripadvisor_hotel_reviews.csv"])
    if csv is None:
        csv = _kagglehub_download(meta["id"], want="tripadvisor_hotel_reviews.csv")

    raw = pd.read_csv(csv, nrows=n_rows)
    df = pd.DataFrame({"text": raw["Review"], "score": raw["Rating"]})
    df = _finalize(df, "tripadvisor", meta["score_scale"][1])
    df.to_parquet(cache, index=False)
    return df


# ==========================================================================
# 4) HRAST (ABSA)
# ==========================================================================
def load_hrast(use_cache: bool = True) -> pd.DataFrame:
    """
    HRAST — Hotel Reviews: Aspects, Sentiments and Topics (23,113 sentences).

    NOTE: the exact column names have to be CHECKED after downloading — so
    this function does not force the canonical schema, it returns the raw
    table and prints the column names. The ABSA code is adapted to it in
    step 05 of the notebook.
    """
    meta = DATASETS["hrast"]
    cache = _cache_path("hrast")

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    csv = _find_local_csv(["hrast.csv", "HRAST.csv"], any_csv_dir="hrast")
    if csv is None:
        csv = _kagglehub_download(meta["id"], want=None)

    raw = pd.read_csv(csv)
    log.warning("HRAST column names (check them and adapt the ABSA code!): %s",
                list(raw.columns))
    raw.to_parquet(cache, index=False)
    return raw


# ==========================================================================
# Download helpers
# ==========================================================================
def _find_local_csv(names: list[str], any_csv_dir: str | None = None) -> Path | None:
    """Check whether the file was downloaded manually into data/raw/."""
    PATHS.ensure()
    for name in names:
        p = PATHS.raw / name
        if p.exists():
            log.info("Found local file: %s", p)
            return p
    if any_csv_dir:
        d = PATHS.raw / any_csv_dir
        if d.is_dir():
            csvs = sorted(d.glob("*.csv"))
            if csvs:
                return csvs[0]
    return None


def _kagglehub_download(dataset_id: str, want: str | None) -> Path:
    """
    Download from Kaggle.

    Prerequisite: a Kaggle API token. See KORNYEZET.md, point 6.
    If it does not work, simply download it manually in the browser and put
    the CSV into the data/raw/ folder — the code finds it automatically.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "'kagglehub' is missing. Install it (pip install kagglehub), OR "
            f"download the '{dataset_id}' dataset manually into the data/raw/ folder."
        ) from e

    log.info("Downloading from Kaggle: %s", dataset_id)
    folder = Path(kagglehub.dataset_download(dataset_id))

    if want:
        target = folder / want
        if target.exists():
            return target
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in: {folder}")
    return csvs[0]


# ==========================================================================
# Saving and reading back the split data
# ==========================================================================
# Notebook 01 saves them, notebooks 02 and 03 read them. In parquet format,
# because it is fast and small — but if pyarrow is not available for some
# reason, it falls back to CSV. This way the pipeline does not stop even if
# the environment is incomplete.

_SPLIT_NAMES = ("train", "val", "test")


def save_splits(splits: dict[str, pd.DataFrame], prefix: str = "booking") -> dict[str, Path]:
    """
    Save the train/val/test sets to the data/processed/ folder.

        save_splits({"train": train_df, "val": val_df, "test": test_df})
    """
    PATHS.ensure()
    written: dict[str, Path] = {}
    for name, part in splits.items():
        pq = PATHS.processed / f"{prefix}_{name}.parquet"
        try:
            part.to_parquet(pq, index=False)
            written[name] = pq
        except Exception:                                         # noqa: BLE001
            csv = PATHS.processed / f"{prefix}_{name}.csv"
            part.to_csv(csv, index=False, encoding="utf-8")
            written[name] = csv
            log.warning("Parquet not available, saved as CSV: %s "
                        "(it would be faster with: pip install pyarrow)", csv.name)
        log.info("%-6s -> %-30s (%d rows)", name, written[name].name, len(part))
    return written


def load_split(name: str, prefix: str = "booking") -> pd.DataFrame:
    """
    Read back one of the sets. Finds it in both parquet and CSV form.

        train_df = load_split("train")
    """
    if name not in _SPLIT_NAMES:
        raise ValueError(f"Unknown split: {name}. Choose from: {_SPLIT_NAMES}")

    pq = PATHS.processed / f"{prefix}_{name}.parquet"
    csv = PATHS.processed / f"{prefix}_{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Neither {pq.name} nor {csv.name} was found.\n"
        "Run the 01_data_loading_eda.ipynb notebook to the end first — "
        "its last cell saves these files."
    )


def load_all_splits(prefix: str = "booking") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train, val, test in one call."""
    return tuple(load_split(n, prefix) for n in _SPLIT_NAMES)   # type: ignore[return-value]


# ==========================================================================
# Single entry point
# ==========================================================================

_LOADERS = {
    "booking_local": load_booking_local,   # data/raw/val.csv — THIS is used now
    "booking_hf": load_booking_hf,
    "kaggle_515k": load_kaggle_515k,
    "tripadvisor": load_tripadvisor,
    "hrast": load_hrast,
}

_BOOKING_SOURCES = {"booking_local", "booking_hf"}


def load_dataset_by_name(name: str, **kwargs) -> pd.DataFrame:
    """
    The external datasets (tripadvisor, kaggle_515k, hrast) can be loaded
    only if CFG.use_external_datasets = True in config.py.
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(_LOADERS)}")
    if name not in _BOOKING_SOURCES and not CFG.use_external_datasets:
        raise PermissionError(
            f"The '{name}' dataset is switched off. For now we work only with "
            "the Booking.com data (CFG.use_external_datasets = False in "
            "config.py). To include it later, set it to True."
        )
    return _LOADERS[name](**kwargs)
