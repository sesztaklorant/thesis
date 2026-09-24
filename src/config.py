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

    # max_length: hány tokenig nézi a modell a szöveget.
    #
    # A val.csv mérése alapján: a reviewk 95%-a 130 szónál rövidebb, a 99%-a
    # 215 szónál. Angolban 1 szó ≈ 1,3 token, tehát a 95. percentilis kb. 170,
    # a 99. kb. 280 token. A korábbi 128 tokenes érték a reviewk jóval több
    # mint 10%-át levágta volna.
    #
    # ⚠ A CSONKOLÁS IRÁNYA FONTOS: a szöveg felépítése
    #       cím . pozitív_rész   negatív_rész
    # tehát a végét levágva SZISZTEMATIKUSAN a NEGATÍV tartalom vész el.
    # Ez a modellt a pozitív osztály felé torzítaná — ezért kell a bőven
    # elégséges max_length. A 03-as notebook külön cellában megméri, hány
    # reviewt érint a csonkolás és torzítja-e az osztályeloszlást.
    max_length: int = 256

    # batch_size: a 4 GB-os RTX 3050 Ti-n 256 tokenes hosszal 4 a biztonságos.
    # A gradient_accumulation_steps=8 visszaállítja a 32-es EFFEKTÍV batchet,
    # így az eredmény matematikailag ugyanaz, csak több lépésben számol.
    batch_size: int = 32          # Colab T4 (16 GB) alapérték
    local_batch_size: int = 4     # saját GPU (4 GB), 256 tokenes hosszal
    local_grad_accum: int = 8     # 4 × 8 = 32 effektív batch
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

    # --- Helyi adatfájl (a Booking hivatalos train/val/test felosztásából) ---
    # A témavezetővel egyeztetve először a ~200k soros val.csv-vel dolgozunk.
    # FONTOS: ez a Booking SAJÁT validációs splitje, nem a mi validációs
    # halmazunk. Lásd a 01-es notebook 2. szakaszát.
    local_csv_name: str = "val.csv"
    source_split_name: str = "booking_val_split"

    def to_json(self, path: Path) -> None:
        """Elmenti a konfigot a kísérlet mellé — így minden eredményhez
        tudni fogod, PONTOSAN milyen beállításokkal készült."""
        d = asdict(self)
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


CFG = Config()


# --------------------------------------------------------------------------
# 3) SZÍNPALETTA — minden ábra ugyanezt használja
# --------------------------------------------------------------------------
# Miért van ez itt és nem minden notebookban külön?
# 1) Konzisztencia: a dolgozat összes ábrája egy rendszerbe tartozik.
# 2) SZÍNLÁTÁSI AKADÁLYMENTESSÉG — ez a fontosabb ok.
#
# A "piros = rossz, zöld = jó" a legelterjedtebb választás, ÉS a lehető
# legrosszabb: a vörös-zöld színtévesztés (deuteranopia/protanopia) a
# férfiak kb. 8%-át érinti, és pont ezt a két színt mossa össze. Egy
# kinyomtatott szakdolgozatnál ez valós olvashatósági probléma.
#
# Helyette PIROS <-> KÉK divergáló pár, semleges szürke középponttal.
# Ez a polaritást (negatív ... pozitív) fejezi ki, és színtévesztéssel is
# elkülöníthető marad. Validálva: a két pólus közti eltérés protanopiával
# szimulálva ΔE 21,6 (a 8-as küszöb bőven felett), normál látással 32,3.
SENTIMENT_COLORS: dict[str, str] = {
    "negative": "#e34948",   # piros
    "neutral":  "#898781",   # semleges szürke (a divergáló skála középpontja)
    "positive": "#2a78d6",   # kék
}

# Egyhues szekvenciális skála nagyságrendhez (hőtérkép, gyakoriság).
# Világostól sötétig — SOHA ne szivárványskála.
SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95",
]

# Divergáló skála "átlaghoz képest" típusú mutatókhoz (pl. neg_lift):
# kék = az átlagnál kevesebb negatív, piros = több. Középen semleges.
DIVERGING_BLUE_RED: list[str] = [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a82c2b",
]

# Ábra-kroma: tengelyek, rácsvonalak, feliratok. Visszafogott, nem versenyez
# az adattal.
INK = {
    "primary":   "#0b0b0b",
    "secondary": "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "axis":      "#c3c2b7",
    "surface":   "#ffffff",
}


def sentiment_palette(order: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """A három sentiment-szín a CFG.labels sorrendjében (vagy megadott sorrendben)."""
    return [SENTIMENT_COLORS[l] for l in (order or CFG.labels)]


def apply_plot_style() -> None:
    """
    Egységes matplotlib-stílus az egész dolgozathoz.
    A notebookok első cellájában hívd meg.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,            # nyomdai minőség
        "savefig.bbox": "tight",
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.spines.top": False,      # visszafogott keret
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "text.color": INK["primary"],
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=["#2a78d6", "#eb6834", "#1baf7a"]),
    })


# --------------------------------------------------------------------------
# 4) ADATHALMAZ-REGISZTER
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
