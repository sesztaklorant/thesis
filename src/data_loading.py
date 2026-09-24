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


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """
    Gyorsítótár írása parquet formátumban.

    A parquet gyors és kicsi, de kell hozzá a pyarrow csomag. Ha valamiért
    nem elérhető, a gyorsítótárazás egyszerűen kimarad — a kód működik
    tovább, csak minden betöltés újraolvassa a CSV-t.
    """
    try:
        df.to_parquet(path, index=False)
        log.info("Gyorsítótárba mentve: %s (%d sor)", path.name, len(df))
    except Exception as e:                                        # noqa: BLE001
        log.warning("Nem sikerült a gyorsítótárazás (%s). A kód működik "
                    "tovább, csak lassabb lesz az újratöltés. "
                    "Megoldás: pip install pyarrow", e.__class__.__name__)


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
# 1/b) Booking.com — HELYI CSV (a hivatalos train/val/test felosztásból)
# ==========================================================================
# A Booking.com adathalmaz három fájlra bontva is letölthető: train.csv,
# val.csv, test.csv. A témavezetővel egyeztetve először a ~200k soros
# val.csv-vel dolgozunk.
#
# ⚠ NÉVÜTKÖZÉS — ezt a dolgozatban is tisztázni kell:
#   A "val.csv" a Booking SAJÁT validációs splitje. Nálunk ez a fájl a
#   TELJES munkaadathalmaz, amit MI osztunk fel 60/20/20 arányban saját
#   train/validation/test halmazra. A kettőnek semmi köze egymáshoz.
#   Ezért nevezzük a betöltött adathalmazt "booking_val_split"-nek, nem
#   "validation"-nek.

# A Booking CSV-k lehetséges oszlopnevei. Több változat is előfordul
# (HF export, GitHub release), ezért aliasokkal dolgozunk.
_BOOKING_ALIASES: dict[str, list[str]] = {
    "review_title":             ["review_title", "title"],
    "review_positive":          ["review_positive", "positive_review", "positive", "pros"],
    "review_negative":          ["review_negative", "negative_review", "negative", "cons"],
    "guest_score":              ["guest_score", "score", "reviewer_score", "rating"],
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

# Ezek a mezők a "Nothing"/"n/a" típusú helykitöltőket tartalmazhatják.
_PLACEHOLDERS = {"nothing", "n/a", "na", "none", "-", "nan", "no negative",
                 "no positive", "nothing to report", "nothing really"}


def _resolve_columns(df_columns) -> dict[str, str]:
    """
    Megkeresi, hogy a CSV melyik tényleges oszlopa felel meg a kanonikus
    mezőknek. Kis-nagybetűre és a szóköz/aláhúzás különbségre érzéketlen.

    Visszatér: {kanonikus_név: tényleges_oszlopnév}
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
    """Helykitöltő szöveg -> üres string."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return "" if s.lower() in _PLACEHOLDERS else s


def load_booking_local(
    filename: str | None = None,
    n_rows: int | None = None,
    combine_pos_neg: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Betölti a data/raw/ mappában lévő Booking.com CSV-t (alapból val.csv).

    Paraméterek
    -----------
    filename : a fájl neve a data/raw/ mappában. None esetén CFG.local_csv_name.
    n_rows   : hány sort olvasson be (None = mind). Fejlesztéshez hasznos.
    combine_pos_neg : a pozitív és negatív szövegrészt egy mezőbe fűzi-e.

    Visszatér
    ---------
    Kanonikus tábla: review_id, text, score, score_scale_max, dataset,
    + a metaadat-oszlopok, amiket a CSV tartalmaz (ország, szállástípus,
    vendégtípus, csillagbesorolás, hónap...). Ezekre épülnek a kereszttáblák.
    """
    filename = filename or CFG.local_csv_name
    cache = _cache_path(f"booking_local_{Path(filename).stem}")

    if use_cache and cache.exists() and n_rows is None:
        try:
            log.info("Gyorsítótárból töltök: %s", cache.name)
            return pd.read_parquet(cache)
        except Exception:                                         # noqa: BLE001
            log.warning("A gyorsítótár nem olvasható, újraolvasom a CSV-t.")

    PATHS.ensure()
    csv_path = PATHS.raw / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nem találom: {csv_path}\n"
            f"Tedd a CSV-t a data/raw/ mappába, vagy add meg a pontos fájlnevet:\n"
            f"    load_booking_local(filename='sajat_nevem.csv')"
        )

    log.info("Olvasás: %s", csv_path)
    raw = pd.read_csv(csv_path, nrows=n_rows, low_memory=False)

    # --- Oszlopok felismerése ---
    cols = _resolve_columns(raw.columns)
    if verbose:
        log.info("Felismert oszlopok (%d/%d):", len(cols), len(_BOOKING_ALIASES))
        for k, v in cols.items():
            log.info("    %-28s <- %s", k, v)
        missing = set(_BOOKING_ALIASES) - set(cols)
        if missing:
            log.warning("Nem találtam: %s", sorted(missing))
            log.warning("A CSV tényleges oszlopai: %s", list(raw.columns))

    if "guest_score" not in cols:
        raise KeyError(
            "Nem találom a pontszám oszlopot. A CSV oszlopai: "
            f"{list(raw.columns)}\nBővítsd a _BOOKING_ALIASES szótárat a "
            "data_loading.py-ban a megfelelő névvel."
        )

    # --- Szöveg összeállítása ---
    def _get(name: str) -> pd.Series:
        if name in cols:
            return raw[cols[name]].map(_clean_placeholder)
        return pd.Series([""] * len(raw), index=raw.index)

    pos, neg, title = _get("review_positive"), _get("review_negative"), _get("review_title")

    if combine_pos_neg:
        text = (title + ". " + pos + " " + neg)
    else:
        text = (pos + " " + neg)
    text = (text.str.replace(r"^\.\s*", "", regex=True)      # üres cím miatti vezető pont
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())

    out = pd.DataFrame({
        "text": text,
        "score": pd.to_numeric(raw[cols["guest_score"]], errors="coerce"),
        "review_positive": pos,
        "review_negative": neg,
    })

    # --- Metaadatok: ezekre épülnek a kereszttáblák ---
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
# A felosztott adat mentése és visszaolvasása
# ==========================================================================
# A 01-es notebook menti, a 02-es és 03-as olvassa. Parquet formátumban,
# mert gyors és kicsi — de ha a pyarrow valamiért nem elérhető, CSV-re vált.
# Így a pipeline akkor sem áll meg, ha a környezet hiányos.

_SPLIT_NAMES = ("train", "val", "test")


def save_splits(splits: dict[str, pd.DataFrame], prefix: str = "booking") -> dict[str, Path]:
    """
    Elmenti a train/val/test halmazokat a data/processed/ mappába.

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
            log.warning("Parquet nem elérhető, CSV-be mentettem: %s "
                        "(gyorsabb lenne: pip install pyarrow)", csv.name)
        log.info("%-6s -> %-30s (%d sor)", name, written[name].name, len(part))
    return written


def load_split(name: str, prefix: str = "booking") -> pd.DataFrame:
    """
    Visszaolvassa az egyik halmazt. Megtalálja parquet és CSV formában is.

        train_df = load_split("train")
    """
    if name not in _SPLIT_NAMES:
        raise ValueError(f"Ismeretlen halmaz: {name}. Választható: {_SPLIT_NAMES}")

    pq = PATHS.processed / f"{prefix}_{name}.parquet"
    csv = PATHS.processed / f"{prefix}_{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Nem találom sem a {pq.name}, sem a {csv.name} fájlt.\n"
        "Futtasd le előbb a 01_data_loading_eda.ipynb notebookot végig — "
        "az utolsó cellája menti ezeket."
    )


def load_all_splits(prefix: str = "booking") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train, val, test egy hívással."""
    return tuple(load_split(n, prefix) for n in _SPLIT_NAMES)   # type: ignore[return-value]


# ==========================================================================
# Egységes belépési pont
# ==========================================================================
_LOADERS = {
    "booking_local": load_booking_local,   # a data/raw/val.csv — EZT használjuk most
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
