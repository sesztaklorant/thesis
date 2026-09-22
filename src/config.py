"""
config.py — A projekt KÖZPONTI beállításai.

Miért van erre szükség?
-----------------------
A szakdolgozatnál a legfontosabb szempont a REPRODUKÁLHATÓSÁG: ha a bíráló
vagy a témavezető megkérdezi, hogy "milyen küszöbértékkel képezted a neutral
osztályt?", akkor egyetlen fájlra tudsz mutatni. Minden "varázsszám"
(random seed, split arányok, küszöbök, elérési utak) ITT lakik, és sehol
máshol nincs beégetve a kódba.

Használat bármelyik notebookból vagy modulból:

    from src.config import CFG, PATHS
    print(CFG.random_seed)
    print(PATHS.raw)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# --------------------------------------------------------------------------
# 1) ELÉRÉSI UTAK
# --------------------------------------------------------------------------
# A projekt gyökere = ennek a fájlnak a szülőmappájának a szülője.
# Így akkor is működik, ha notebookból, ha szkriptből futtatod.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    # adat
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"          # érintetlen letöltés
    interim: Path = PROJECT_ROOT / "data" / "interim"  # félkész, tisztított
    processed: Path = PROJECT_ROOT / "data" / "processed"  # modellezésre kész

    # eredmények
    results: Path = PROJECT_ROOT / "results"
    figures: Path = PROJECT_ROOT / "results" / "figures"   # ábrák a dolgozatba
    tables: Path = PROJECT_ROOT / "results" / "tables"     # táblázatok (csv)
    models: Path = PROJECT_ROOT / "results" / "models"     # mentett modellek

    def ensure(self) -> None:
        """Létrehozza az összes mappát, ha még nem létezik."""
        for p in [
            self.data, self.raw, self.interim, self.processed,
            self.results, self.figures, self.tables, self.models,
        ]:
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# --------------------------------------------------------------------------
# 2) KÍSÉRLETI BEÁLLÍTÁSOK
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Reprodukálhatóság ---
    # Minden véletlent használó műveletnél (split, model init, shuffle)
    # ezt adjuk át. Ettől lesz újrafuttatva UGYANAZ az eredmény.
    random_seed: int = 42

    # --- Adatfelosztás (a Milestone 1 szerint 60/20/20) ---
    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    # Rétegzett (stratified) split: minden halmazban ugyanaz az osztályarány.
    # 3 osztályos, kiegyensúlyozatlan feladatnál ez KÖTELEZŐ.
    stratify: bool = True

    # --- Sentiment címkézés a vendégértékelésből ---
    # A Booking.com / 515K adathalmazokban NINCS kész sentiment címke,
    # csak egy 1–10-es pontszám. Ebből kell 3 osztályt képezni.
    #
    #   score <= neg_threshold           -> negative
    #   neg_threshold < score < pos_thr  -> neutral
    #   score >= pos_threshold           -> positive
    #
    # FONTOS: ezt a döntést a dolgozatban INDOKOLNI kell (lásd MUNKATERV.md,
    # 3. fázis). A 6/8 vágás a szakirodalomban gyakori, de érzékenységi
    # vizsgálatot (pl. 5/8 és 7/9 változattal) érdemes mellé tenni.
    score_scale_max: float = 10.0
    neg_threshold: float = 6.0
    pos_threshold: float = 8.0

    # A három osztály kanonikus sorrendje. MINDENHOL ezt használjuk,
    # hogy a confusion matrix tengelyei konzisztensek legyenek.
    labels: tuple[str, ...] = ("negative", "neutral", "positive")

    # --- Mintavételezés (fejlesztés közben) ---
    # 1.6 millió soron nem fejlesztünk. Először kis mintán megy a pipeline,
    # és csak a végleges futtatásnál kapcsoljuk ki a mintavételt.
    dev_sample_size: int | None = 50_000   # None = teljes adathalmaz
    # Osztály-kiegyensúlyozás: a neutral osztály jellemzően kisebbségi.
    # "downsample" = a nagy osztályokat csökkentjük a legkisebb méretére.
    balance_strategy: str = "none"  # "none" | "downsample"

    # --- Szövegelőkészítés ---
    min_chars: int = 10         # ennél rövidebb szöveg zajnak számít
    max_chars: int = 5_000      # extrém hosszú kiugró értékek levágása
    drop_duplicates: bool = True
    keep_english_only: bool = True

    # --- TF-IDF (klasszikus ML) ---
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)   # unigram + bigram
    tfidf_min_df: int = 3        # ennél ritkább szó kimarad (zajszűrés)
    tfidf_max_df: float = 0.9    # a dokumentumok >90%-ában előforduló szó kimarad
    tfidf_sublinear_tf: bool = True

    # --- Transformer (BERT) ---
    model_name: str = "bert-base-uncased"
    max_length: int = 128        # tokenben; hotel reviewkhoz 128 általában elég
    batch_size: int = 32
    learning_rate: float = 2e-5  # a BERT-papír ajánlása: 2e-5 / 3e-5 / 5e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # --- ABSA aspektusok (HRAST + kulcsszavas alapmegoldás) ---
    aspects: tuple[str, ...] = (
        "room", "staff", "cleanliness", "facilities", "location",
        "breakfast", "pool", "bathroom", "parking", "noise",
        "wifi", "value_for_money",
    )

    def to_json(self, path: Path) -> None:
        """Elmenti a konfigot a kísérlet mellé — így minden eredményhez
        tudni fogod, PONTOSAN milyen beállításokkal készült."""
        d = asdict(self)
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


CFG = Config()


# --------------------------------------------------------------------------
# 3) ADATHALMAZ-REGISZTER
# --------------------------------------------------------------------------
# Egy helyen tartjuk, hogy melyik adathalmaz honnan jön és mi a szerepe.
# A dolgozat "Data Sources" fejezetét ebből tudod majd kitölteni.
DATASETS = {
    "booking_hf": {
        "role": "main",  # fő sentiment-klasszifikáció (RQ1, RQ2)
        "source": "huggingface",
        "id": "Booking-com/accommodation-reviews",
        "license": "CC BY-SA 4.0 (nem kereskedelmi)",
        "size": "~1.6M review, ~40k szálláshely",
        "text_fields": ["review_title", "review_positive", "review_negative"],
        "score_field": "guest_score",
        "score_scale": (1.0, 10.0),
    },
    "kaggle_515k": {
        "role": "external_validation",  # RQ5 — cross-dataset általánosítás
        "source": "kaggle",
        "id": "jiashenliu/515k-hotel-reviews-data-in-europe",
        "license": "Kaggle — a leírásban megadott feltételek (oktatási célra)",
        "size": "~515k review, 1493 európai hotel",
        "text_fields": ["Positive_Review", "Negative_Review"],
        "score_field": "Reviewer_Score",
        "score_scale": (2.5, 10.0),
    },
    "tripadvisor": {
        "role": "external_validation_alt",  # MÁS platform -> erősebb RQ5 teszt
        "source": "kaggle",
        "id": "andrewmvd/trip-advisor-hotel-reviews",
        "license": "CC BY-NC-SA 4.0",
        "size": "~20k review",
        "text_fields": ["Review"],
        "score_field": "Rating",
        "score_scale": (1.0, 5.0),
    },
    "hrast": {
        "role": "absa",  # RQ3, RQ4 — aspektus-szintű annotáció
        "source": "kaggle",
        "id": "costastziouvas/hotel-reviews-aspects-sentiments-and-topics",
        "license": "ellenőrizd a Kaggle oldalon és dokumentáld!",
        "size": "23 113 annotált mondat",
        "text_fields": ["sentence"],  # pontos oszlopnév letöltés után ellenőrizendő
        "score_field": None,
    },
}
