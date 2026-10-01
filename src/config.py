"""
config.py — A projekt KÖZPONTI beállításai / CENTRAL settings of the project.

HU: Minden "mágikus szám" (seed, küszöbök, modellnevek, útvonalak) itt van, és
    sehol máshol nincs beégetve. Így a dolgozatban minden eredményhez meg lehet
    mondani, pontosan milyen beállítás hozta létre.
EN: Every "magic number" (seed, thresholds, model names, paths) lives here and
    nowhere else. This way every result in the thesis can be traced back to
    the exact settings that produced it.

    from src.config import CFG, PATHS
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# 1) ÚTVONALAK / PATHS
# --------------------------------------------------------------------------
# HU: Projekt gyökér = ennek a fájlnak a szülőmappájának a szülője.
# EN: Project root = the parent of this file's parent folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"              # HU: érintetlen letöltés / EN: untouched download
    interim: Path = PROJECT_ROOT / "data" / "interim"      # HU: köztes, tisztított / EN: intermediate, cleaned
    processed: Path = PROJECT_ROOT / "data" / "processed"  # HU: modellezésre kész / EN: ready for modelling
    labels: Path = PROJECT_ROOT / "data" / "labels"        # HU: BERT-címkék / EN: BERT-generated labels
    annotation: Path = PROJECT_ROOT / "data" / "annotation"  # HU: kézi gold minta / EN: manual gold sample

    results: Path = PROJECT_ROOT / "results"
    figures: Path = PROJECT_ROOT / "results" / "figures"
    tables: Path = PROJECT_ROOT / "results" / "tables"
    models: Path = PROJECT_ROOT / "results" / "models"

    def ensure(self) -> None:
        """HU: Minden mappát létrehoz, ha még nincs. / EN: Create every folder if missing."""
        for p in (self.data, self.raw, self.interim, self.processed, self.labels,
                  self.annotation, self.results, self.figures, self.tables, self.models):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# --------------------------------------------------------------------------
# 2) BEÁLLÍTÁSOK / SETTINGS
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Reprodukálhatóság / Reproducibility ---
    random_seed: int = 42

    # --- Adatforrás / Data source ---
    # HU: A Booking.com hivatalos felosztásának val.csv fájlja (~200k sor). Ez
    #     NEKÜNK a teljes munkaadat — nem keverendő a saját validációs halmazzal.
    # EN: val.csv of Booking.com's official split (~200k rows). For US it is the
    #     whole working dataset — not to be confused with our own validation set.
    local_csv_name: str = "val.csv"
    source_split_name: str = "booking_val_split"

    # --- Osztályok / Classes ---
    # HU: Kanonikus sorrend, MINDENHOL ezt használjuk (valószínűség-oszlopok,
    #     konfúziós mátrixok tengelyei).
    # EN: Canonical order, used EVERYWHERE (probability columns, confusion axes).
    labels: tuple[str, ...] = ("negative", "neutral", "positive")

    # --- Szöveg-előkészítés / Text preparation ---
    min_chars: int = 10
    max_chars: int = 5_000
    drop_duplicates: bool = True
    keep_english_only: bool = True

    # --- Fejlesztési mintavétel / Development sampling ---
    # HU: A BERT-címkézés 200k soron GPU-n is ~15–30 perc modellenként. Fejlesztés
    #     közben kis mintán futtatunk; a végső futásnál None = teljes adat.
    # EN: BERT labelling on 200k rows takes ~15–30 min per model even on a GPU.
    #     During development we run on a small sample; None = full data.
    labelling_sample_size: int | None = 20_000

    # ======================================================================
    # BERT-ALAPÚ CÍMKÉZÉS / BERT-BASED LABELLING
    # ======================================================================
    # HU: Közös inferencia-beállítások mindhárom megközelítéshez.
    # EN: Shared inference settings for all three approaches.
    #
    # HU: max_length: a val.csv-n mérve 256 tokennél a vélemények ~1,4%-a
    #     csonkolódik. A szöveg "cím. pozitív rész negatív rész" felépítésű,
    #     így a csonkolás a NEGATÍV részt vágja le először — ezért nem 128.
    # EN: max_length: measured on val.csv, 256 tokens truncate ~1.4% of the
    #     reviews. The text is "title. positive part negative part", so
    #     truncation cuts the NEGATIVE part first — hence not 128.
    max_length: int = 256
    inference_batch_size: int = 64        # HU: 4 GB VRAM-on fp16-tal biztonságos / EN: safe on 4 GB VRAM with fp16
    inference_chunk_size: int = 10_000    # HU: ennyi soronként mentünk (folytatható) / EN: checkpoint every N rows (resumable)

    # --- (A) Zero-shot: kész, már finomhangolt modellek / ready fine-tuned models ---
    # HU: 3 osztályos modell (negatív / semleges / pozitív). RoBERTa-alapú (BERT-család),
    #     tweeteken tanítva — a doménkülönbséget a dolgozatban jelezni kell.
    # EN: 3-class model (negative / neutral / positive). RoBERTa-based (BERT family),
    #     trained on tweets — the domain gap has to be stated in the thesis.
    zero_shot_3class_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # HU: 1–5 csillagos BERT-modell termékvéleményeken tanítva. Ebből képezzük az
    #     1–10-es pontszámot (várható csillagszám lineáris átskálázásával).
    # EN: 1–5 star BERT model trained on product reviews. The 1–10 score is
    #     derived from it (linear rescaling of the expected number of stars).
    zero_shot_stars_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"

    # --- (B) Transzfertanulás HRAST-on / Transfer learning on HRAST ---
    # HU: Kézzel annotált, mondatszintű hotel-adat. Itt CSAK tanítóadatként
    #     szerepel a címkéző modellhez — a Booking-elemzés továbbra is csak Booking.
    # EN: Human-annotated, sentence-level hotel data. Used ONLY as training data
    #     for the labeller — the analysis itself stays Booking-only.
    hrast_base_model: str = "bert-base-uncased"
    hrast_val_size: float = 0.15
    hrast_test_size: float = 0.15
    # HU: Ha egy mondatban pozitív ÉS negatív aspektus is van: "drop" = kihagyjuk,
    #     "neutral" = semlegesnek vesszük.
    # EN: If a sentence has both a positive AND a negative aspect: "drop" = skip,
    #     "neutral" = treat as neutral.
    hrast_conflict_policy: str = "drop"
    # HU: B-t mondatszinten alkalmazzuk (mint ahogy tanult), majd véleményre aggregálunk.
    # EN: B is applied at sentence level (as it was trained), then aggregated.
    hrast_granularity: str = "sentence"

    # --- (C) Önképzés / Self-training (pseudo-labelling) on Booking ---
    self_training_base_model: str = "bert-base-uncased"
    # HU: Csak a tanár(ok) által legalább ilyen biztonsággal címkézett vélemények
    #     kerülnek a tanítóhalmazba.
    # EN: Only reviews labelled by the teacher(s) with at least this confidence
    #     enter the training set.
    pseudo_label_min_confidence: float = 0.80
    # HU: True = csak ott, ahol A és B egyetért (ha B elérhető).
    # EN: True = only where A and B agree (if B is available).
    pseudo_label_require_agreement: bool = True
    # HU: Osztályonként legfeljebb ennyi példa (kiegyensúlyozás + futási idő).
    # EN: At most this many examples per class (balancing + running time).
    pseudo_label_max_per_class: int = 15_000

    # --- Finomhangolás / Fine-tuning (B and C) ---
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 32            # HU: Colab T4 (16 GB) / EN: Colab T4 (16 GB)
    local_batch_size: int = 4       # HU: saját GPU (4 GB) / EN: own GPU (4 GB)
    local_grad_accum: int = 8       # HU: 4 × 8 = 32 effektív batch / EN: 4 × 8 = 32 effective batch
    small_gpu_vram_gb: float = 8.0  # HU: ez alatt a "local" beállítás él / EN: below this the "local" setting applies

    # --- Végső címke / Final label ---
    # HU: A három megközelítés többségi szavazata adja a végső címkét.
    #     Döntetlennél ez a megközelítés dönt (a gold minta alapján érdemes
    #     átírni a legjobbra).
    # EN: The final label is the majority vote of the three approaches.
    #     On a tie this approach decides (update it to the best one after the
    #     gold-sample evaluation).
    tie_breaker_approach: str = "A"

    # --- Kézi annotáció / Manual annotation (gold standard) ---
    gold_sample_size: int = 400

    # --- Csak REFERENCIA: pontszám-vágás / REFERENCE ONLY: score cut ---
    # HU: A konzulens jogosan jelezte, hogy a vendégpontszám vágása NEM érvényes
    #     szentiment-címke. Csak összehasonlításra tartjuk meg: megmutatja,
    #     mennyire térne el a naiv vágás a szövegalapú címkéktől.
    # EN: As the supervisor rightly noted, cutting the guest score is NOT a valid
    #     sentiment label. Kept for comparison only: it shows how far the naive
    #     cut would deviate from the text-based labels.
    reference_neg_threshold: float = 7.0
    reference_pos_threshold: float = 9.0

    # --- TF-IDF (klasszikus modellek / classical models) ---
    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.9
    tfidf_sublinear_tf: bool = True

    # --- Aspektusok (kulcsszavas) / Aspects (keyword-based) ---
    aspects: tuple[str, ...] = (
        "room", "staff", "cleanliness", "facilities", "location",
        "breakfast", "pool", "bathroom", "parking", "noise",
        "wifi", "value_for_money",
    )

    def to_json(self, path: Path) -> None:
        """HU: A beállítások mentése az eredmények mellé. / EN: Save the settings next to the results."""
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


CFG = Config()

# HU: A három címkéző megközelítés rövid neve -> leírás (ábrák, táblák).
# EN: Short name of the three labelling approaches -> description (figures, tables).
APPROACHES: dict[str, str] = {
    "A": "Zero-shot (pre-trained sentiment models)",
    "B": "Transfer learning (BERT fine-tuned on HRAST)",
    "C": "Self-training (BERT fine-tuned on Booking pseudo-labels)",
}


# --------------------------------------------------------------------------
# 3) SZÍNPALETTA / COLOUR PALETTE
# --------------------------------------------------------------------------
# HU: Piros–szürke–kék (NEM piros–zöld): színtévesztők is meg tudják
#     különböztetni, és nyomtatásban is olvasható.
# EN: Red–grey–blue (NOT red–green): distinguishable for colour-blind readers
#     and readable in print.
SENTIMENT_COLORS: dict[str, str] = {
    "negative": "#e34948",
    "neutral":  "#898781",
    "positive": "#2a78d6",
}

APPROACH_COLORS: dict[str, str] = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#1baf7a",
    "final": "#52514e",
    "score_cut": "#898781",
}

SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95",
]

DIVERGING_BLUE_RED: list[str] = [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a82c2b",
]

INK = {
    "primary":   "#0b0b0b",
    "secondary": "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "axis":      "#c3c2b7",
    "surface":   "#ffffff",
}


def sentiment_palette(order: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """HU: A három szentiment-szín CFG.labels sorrendben. / EN: The three sentiment colours in CFG.labels order."""
    return [SENTIMENT_COLORS[label] for label in (order or CFG.labels)]


def apply_plot_style() -> None:
    """HU: Egységes matplotlib-stílus a dolgozat minden ábrájához.
    EN: Uniform matplotlib style for every figure of the thesis."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "text.color": INK["primary"],
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=[APPROACH_COLORS["A"], APPROACH_COLORS["B"],
                                             APPROACH_COLORS["C"]]),
    })


# --------------------------------------------------------------------------
# 4) ADATFORRÁSOK / DATA SOURCES
# --------------------------------------------------------------------------
# HU: A "Felhasznált technológiák / adatok" fejezet innen tölthető ki.
# EN: The "Technologies / data used" chapter can be filled in from here.
DATASETS = {
    "booking": {
        "role": "analysed data (unlabelled for sentiment)",
        "source": "Booking.com Accommodation Reviews (official val split)",
        "id": "Booking-com/accommodation-reviews",
        "license": "CC BY-SA 4.0 (non-commercial)",
        "size": "~200k reviews in val.csv",
        "text_fields": ["review_title", "review_positive", "review_negative"],
        "score_field": "guest_score",
        "score_scale": (1.0, 10.0),
    },
    "hrast": {
        "role": "training data for labelling approach B only",
        "source": "kaggle",
        "id": "costastziouvas/hotel-reviews-aspects-sentiments-and-topics",
        "license": "check on the Kaggle page and document it",
        "size": "23,113 annotated sentences",
    },
}
