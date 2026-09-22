"""
data_loading.py — Adatletöltés és egységes formára hozás.

A három adathalmaz nagyon különböző szerkezetű. A cél az, hogy mindegyikből
UGYANAZT a táblát állítsuk elő, hogy a további kód (tisztítás, TF-IDF,
BERT) ne tudja, honnan jött az adat:

    review_id | text | score | score_scale_max | dataset | (opcionális extra oszlopok)

Így az RQ5 (cross-dataset általánosítás) kísérlet triviális lesz: ugyanazt a
függvényt hívod más `dataset` paraméterrel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import CFG, PATHS, DATASETS

log = logging.getLogger(__name__)

# Az egységes séma, amit minden loader visszaad.
CANONICAL_COLUMNS = ["review_id", "text", "score", "score_scale_max", "dataset"]


# ==========================================================================
# Segédfüggvények
# ==========================================================================
def _cache_path(name: str) -> Path:
    """Hova mentjük a letöltött, még nyers adatot."""
    PATHS.ensure()
    return PATHS.raw / f"{name}.parquet"


def _finalize(df: pd.DataFrame, dataset: str, scale_max: float) -> pd.DataFrame:
    """Egységes oszlopnevek + típusok + sorszámozott azonosító."""
    df = df.copy()
    df["dataset"] = dataset
    df["score_scale_max"] = float(scale_max)
    df["review_id"] = [f"{dataset}_{i}" for i in range(len(df))]
    df["text"] = df["text"].astype("string")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    return df[CANONICAL_COLUMNS + extra]


# ==========================================================================
# 1) Booking.com — Hugging Face (FŐ adathalmaz)
# ==========================================================================
def load_booking_hf(
    n_rows: int | None = None,
    use_cache: bool = True,
    combine_pos_neg: bool = True,
) -> pd.DataFrame:
    """
    Booking.com Accommodation Reviews (CC BY-SA 4.0).

    Paraméterek
    -----------
    n_rows : hány sort tölts be (None = mind ~1.6M). Fejlesztéshez adj meg
             pl. 50_000-et, különben sokáig tart és sok memóriát eszik.
    combine_pos_neg : True esetén a `review_positive` és `review_negative`
             mezőt egyetlen szöveggé fűzzük. Ez felel meg a dolgozatban
             leírt "review-szintű" osztályozásnak.

             FIGYELEM (ezt érdemes a dolgozatban is megemlíteni): a két mező
             külön-külön már magában hordoz egy sentiment-címkét. Ha együtt
             használod őket ÉS a pontszámból képzed a címkét, az konzisztens.
             De ha külön mondatokra bontod őket, akkor a "positive" mezőből
             származó mondatok szinte definíció szerint pozitívak — ez nem
             hiba, csak dokumentálni kell.
    """
    meta = DATASETS["booking_hf"]
    cache = _cache_path("booking_hf")

    if use_cache and cache.exists():
        log.info("Gyorsítótárból töltök: %s", cache)
        df = pd.read_parquet(cache)
        return df.head(n_rows) if n_rows else df

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Hiányzik a 'datasets' csomag. Telepítsd: pip install datasets"
        ) from e

    log.info("Letöltés a Hugging Face-ről: %s", meta["id"])
    # streaming=True: nem tölti le az egész 1.6M sort a memóriába, soronként ad.
    # Fejlesztéskor ez a barátod.
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

    # --- Szöveg összeállítása ---
    def _clean_part(s):
        """A Booking adatban a hiányzó rész gyakran 'Nothing' / üres string."""
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
        # Hasznos extra mezők az EDA-hoz és a hibaelemzéshez:
        "accommodation_type": raw.get("accommodation_type"),
        "accommodation_country": raw.get("accommodation_country"),
        "guest_type": raw.get("guest_type"),
        "review_positive": pos,
        "review_negative": neg,
    })

    df = _finalize(df, "booking_hf", meta["score_scale"][1])
    df.to_parquet(cache, index=False)
    log.info("Elmentve: %s (%d sor)", cache, len(df))
    return df


# ==========================================================================
# 2) Kaggle 515K Hotel Reviews (külső validáció)
# ==========================================================================
def load_kaggle_515k(n_rows: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    515K Hotel Reviews Data in Europe.

    Letöltés: kagglehub-bal automatikusan, VAGY kézzel a Kaggle-ről
    (Hotel_Reviews.csv) a data/raw/ mappába.
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
# 3) TripAdvisor (ALTERNATÍV külső validáció — MÁS platform)
# ==========================================================================
def load_tripadvisor(n_rows: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    TripAdvisor Hotel Reviews (~20k, 1–5 csillag).

    Miért fontos? A Booking.com és az 515K adathalmaz UGYANARRÓL a
    platformról származik, csak más időszakból. Ha a "cross-dataset
    generalization" (RQ5) kérdést komolyan akarod megválaszolni, egy
    másik platform adata sokkal erősebb tesztet ad.
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
    HRAST — Hotel Reviews: Aspects, Sentiments and Topics (23 113 mondat).

    FIGYELEM: a pontos oszlopneveket a letöltés után ELLENŐRIZNI kell —
    ez a függvény ezért nem erőlteti a kanonikus sémát, hanem visszaadja
    a nyers táblát, és kiírja az oszlopneveket. A notebook 05-ös lépésében
    igazítjuk hozzá az ABSA kódot.
    """
    meta = DATASETS["hrast"]
    cache = _cache_path("hrast")

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    csv = _find_local_csv(["hrast.csv", "HRAST.csv"], any_csv_dir="hrast")
    if csv is None:
        csv = _kagglehub_download(meta["id"], want=None)

    raw = pd.read_csv(csv)
    log.warning("HRAST oszlopnevek (ellenőrizd és igazítsd az ABSA kódot!): %s",
                list(raw.columns))
    raw.to_parquet(cache, index=False)
    return raw


# ==========================================================================
# Letöltési segédek
# ==========================================================================
def _find_local_csv(names: list[str], any_csv_dir: str | None = None) -> Path | None:
    """Megnézi, hogy kézzel letöltötted-e a fájlt a data/raw/ mappába."""
    PATHS.ensure()
    for name in names:
        p = PATHS.raw / name
        if p.exists():
            log.info("Helyi fájlt találtam: %s", p)
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
    Letöltés Kaggle-ről.

    Előfeltétel: Kaggle API token. Lásd KORNYEZET.md 6. pont.
    Ha nem megy, egyszerűen töltsd le kézzel a böngészőből, és tedd a
    CSV-t a data/raw/ mappába — a kód automatikusan megtalálja.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "Hiányzik a 'kagglehub'. Telepítsd (pip install kagglehub), VAGY "
            f"töltsd le kézzel a(z) '{dataset_id}' adathalmazt a data/raw/ mappába."
        ) from e

    log.info("Letöltés Kaggle-ről: %s", dataset_id)
    folder = Path(kagglehub.dataset_download(dataset_id))

    if want:
        target = folder / want
        if target.exists():
            return target
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nem találtam CSV-t itt: {folder}")
    return csvs[0]


# ==========================================================================
# Egységes belépési pont
# ==========================================================================
_LOADERS = {
    "booking_hf": load_booking_hf,
    "kaggle_515k": load_kaggle_515k,
    "tripadvisor": load_tripadvisor,
    "hrast": load_hrast,
}


def load_dataset_by_name(name: str, **kwargs) -> pd.DataFrame:
    """
    Példa:
        df_train_source = load_dataset_by_name("booking_hf", n_rows=50_000)
        df_external     = load_dataset_by_name("tripadvisor")
    """
    if name not in _LOADERS:
        raise ValueError(f"Ismeretlen adathalmaz: {name}. Választható: {list(_LOADERS)}")
    return _LOADERS[name](**kwargs)
