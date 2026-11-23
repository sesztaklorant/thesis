"""
Central settings of the project.

Paths, model names, thresholds and other constants are defined only here, so every
result can be traced back to the settings that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# 1) PATHS
# --------------------------------------------------------------------------
# Project root: two levels above this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"  # downloaded datasets, not modified
    raw_booking: Path = PROJECT_ROOT / "data" / "raw" / "booking"  # Booking.com val.csv
    raw_hrast: Path = PROJECT_ROOT / "data" / "raw" / "hrast"  # HRAST csv
    interim: Path = PROJECT_ROOT / "data" / "interim"  # cached intermediate files
    processed: Path = PROJECT_ROOT / "data" / "processed"  # ready for modelling
    labels: Path = PROJECT_ROOT / "data" / "labels"  # BERT-generated labels
    annotation: Path = PROJECT_ROOT / "data" / "annotation"  # manual gold sample

    results: Path = PROJECT_ROOT / "results"
    figures: Path = PROJECT_ROOT / "results" / "figures"
    tables: Path = PROJECT_ROOT / "results" / "tables"
    models: Path = PROJECT_ROOT / "results" / "models"

    def ensure(self) -> None:
        """Create every folder if missing."""
        for p in (self.data, self.raw, self.raw_booking, self.raw_hrast, self.interim,
                  self.processed, self.labels,
                  self.annotation, self.results, self.figures, self.tables, self.models):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# --------------------------------------------------------------------------
# 2) SETTINGS
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Reproducibility ---
    random_seed: int = 42

    # --- Data source ---
    # val.csv of the official Booking.com split (~200k rows). It is the whole
    # working dataset of the project, not the validation set of the models.
    local_csv_name: str = "val.csv"
    source_split_name: str = "booking_val_split"

    # --- Classes ---
    # Class order used everywhere (probability columns, confusion matrix axes).
    labels: tuple[str, ...] = ("negative", "neutral", "positive")

    # --- Text preparation ---
    min_chars: int = 10
    max_chars: int = 5_000
    drop_duplicates: bool = True
    keep_english_only: bool = True

    # --- Development sampling ---
    # BERT labelling of 200k rows takes 15–30 min per model on a GPU, so development
    # used a sample. None = full data.
    labelling_sample_size: int | None = None  # 20_000 during development

    # ======================================================================
    # BERT-BASED LABELLING
    # ======================================================================
    # Inference settings shared by the three approaches.
    # 256 tokens truncate about 1.4% of the reviews. The text is "title. praise complaint",
    # so a shorter limit would cut the complaint part first.
    max_length: int = 256
    inference_batch_size: int = 64  # safe on 4 GB VRAM with fp16
    inference_chunk_size: int = 10_000  # checkpoint every N rows (resumable)

    # --- (A) Zero-shot: ready fine-tuned models ---
    # 3-class model (negative / neutral / positive), RoBERTa-based, trained on tweets.
    zero_shot_3class_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # 1–5 star model trained on product reviews. The 1–10 score is a linear
    # rescaling of the expected number of stars.
    zero_shot_stars_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"

    # --- (B) Transfer learning on HRAST ---
    # Human-annotated hotel sentences, used only to train labeller (B).
    hrast_base_model: str = "bert-base-uncased"
    hrast_val_size: float = 0.15
    hrast_test_size: float = 0.15
    # If a sentence has both a positive and a negative aspect: "drop" = skip,
    # "neutral" = treat as neutral.
    hrast_conflict_policy: str = "drop"
    # (B) is applied to sentences, as it was trained, and aggregated per review.
    hrast_granularity: str = "sentence"

    # --- (C) Self-training (pseudo-labelling) on Booking ---
    self_training_base_model: str = "bert-base-uncased"
    # Minimum teacher confidence for a review to enter the training set.
    pseudo_label_min_confidence: float = 0.80
    # True = only reviews where (A) and (B) agree.
    pseudo_label_require_agreement: bool = True
    # Per-class cap (class balance and running time).
    pseudo_label_max_per_class: int = 15_000

    # --- Fine-tuning (B and C) ---
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 32  # GPUs with 8 GB or more
    local_batch_size: int = 4  # own GPU (4 GB)
    local_grad_accum: int = 8  # 4 × 8 = 32 effective batch
    small_gpu_vram_gb: float = 8.0  # below this the "local" setting applies

    # --- Neutral band (review level) ---
    # Almost half of the reviews are mixed (praise and complaint), and the argmax of the
    # averaged sentence probabilities labels them positive. With the band rule a review is
    # neutral if |P(pos) − P(neg)| < band, which keeps the label consistent with the 1–10 score.
    review_neutral_band: float = 0.25  # calibrated on the gold sample

    # --- Final label ---
    # Approaches that form the final label. (B) performed best on the gold sample, so the
    # final label is (B); the (A) and (C) columns are kept for comparison.
    # ("A", "B", "C") would mean a majority vote with tie_breaker_approach for ties.
    final_label_approaches: tuple[str, ...] = ("B",)
    tie_breaker_approach: str = "B"

    # ======================================================================
    # LAYER 2: THEMATIC CATEGORISATION
    # ======================================================================
    # --- 2/a: aspect classifier trained on HRAST ---
    # Multi-label BERT: a sentence can belong to several aspects.
    aspect_base_model: str = "bert-base-uncased"
    aspect_max_length: int = 128  # sentences, not reviews
    aspect_val_size: float = 0.15
    aspect_test_size: float = 0.15
    aspect_min_examples: int = 50  # rarer aspects are dropped
    aspect_max_pos_weight: float = 10.0  # cap on rare-aspect weight

    # --- 2/b: subtopics (BERTopic approach) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    topic_min_cluster_size: int = 40  # smaller clusters are not topics
    topic_min_samples: int = 10
    # Maximum number of sentences per aspect used to fit the subtopic model; the remaining
    # sentences are assigned to the nearest subtopic. With 15,000 sentences HDBSCAN returned
    # one giant cluster for several aspects on the full data (negative staff: 97%).
    topic_sample_per_aspect: int = 5_000
    topic_min_sentences: int = 300  # no subtopics below this
    topic_outlier_threshold: float = 0.45  # reassign noise if cosine ≥ this
    topic_top_n_terms: int = 10
    topic_stability_seeds: tuple[int, ...] = (42, 7, 123)
    # Subtopics come from fine clustering followed by merging of the most similar topics,
    # until at most subtopic_max_topics remain and no two topics are more similar than
    # subtopic_merge_similarity. The earlier versions are listed in docs/DEVELOPMENT_LOG.md.
    topic_overview_min_share: float = 0.01
    subtopic_min_cluster_share: float = 0.02
    subtopic_max_topics: int = 10
    subtopic_merge_similarity: float = 0.90
    # Upper limit for a merged subtopic, which prevents catch-all topics.
    subtopic_max_topic_share: float = 0.35
    # A cluster above this share triggers re-clustering with half the cluster size.
    subtopic_max_initial_share: float = 0.50
    subtopic_sweep_shares: tuple[float, ...] = (0.01, 0.02)
    subtopic_sweep_max_topics: tuple = (None, 12, 10, 8, 6)  # None = no merging
    heatmap_max_subtopics: int = 8  # max subtopics on a complaint map
    heatmap_min_effect: float = 0.20  # marked only if |lift − 1| also reaches this
    # Version tag of the subtopic run. It is part of every 2/b output name, so a new
    # setting does not overwrite earlier tables, figures or manual checks.
    topic_version: str = "full2"
    # Comma-separated lists in the praise field are split into separate units
    # ("Great location, friendly staff, clean rooms" -> 3 units).
    split_praise_lists: bool = True

    # --- Manual annotation (gold standard) ---
    gold_sample_size: int = 400

    # --- Reference only: score cut ---
    # A cut of the guest score is not a valid sentiment label. It is kept to show
    # how far such a cut deviates from the text-based labels.
    reference_neg_threshold: float = 7.0
    reference_pos_threshold: float = 9.0

    # --- TF-IDF (classical models) ---
    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.9
    tfidf_sublinear_tf: bool = True

    # --- Aspects (keyword-based) ---
    aspects: tuple[str, ...] = (
        "room", "staff", "cleanliness", "facilities", "location",
        "breakfast", "pool", "bathroom", "parking", "noise",
        "wifi", "value_for_money",
    )

    def to_json(self, path: Path) -> None:
        """Save the settings next to the results."""
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


CFG = Config()

# HRAST aspect columns -> aspect names used in the project. The "Aspect" column is left out
# because its meaning is unclear. Names follow the keyword aspects (aspects.py) where possible.
HRAST_ASPECT_MAP: dict[str, str] = {
    "Clean": "cleanliness", "Comfort": "comfort", "Facilities/Amenities": "facilities",
    "Location": "location", "Restaurant (dinner)": "restaurant", "Staff": "staff",
    "View (Balcony)": "view", "Breakfast": "breakfast", "Room": "room", "Pool": "pool",
    "Beach": "beach", "Bathroom/Shower (toilet)": "bathroom", "Bar": "bar", "Bed": "bed",
    "Parking": "parking", "Noise": "noise", "Reception-checkin": "reception", "Lift": "lift",
    "Value for money": "value_for_money", "Wi-Fi": "wifi", "Generic": "generic",
}

# Aspects without subtopics (general, non-thematic sentences).
NON_THEMATIC_ASPECTS: tuple[str, ...] = ("generic",)

# Labelling approaches: short name -> description used in figures and tables.
APPROACHES: dict[str, str] = {
    "A": "Zero-shot (pre-trained sentiment models)",
    "B": "Transfer learning (BERT fine-tuned on HRAST)",
    "C": "Self-training (BERT fine-tuned on Booking pseudo-labels)",
}


# --------------------------------------------------------------------------
# 3) COLOUR PALETTE
# --------------------------------------------------------------------------
# Red-grey-blue instead of red-green, readable for colour-blind readers and in print.
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
    """The three sentiment colours in CFG.labels order."""
    return [SENTIMENT_COLORS[label] for label in (order or CFG.labels)]


def apply_plot_style() -> None:
    """Common matplotlib style for all figures."""
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
# 4) DATA SOURCES
# --------------------------------------------------------------------------
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
        "license": "CC BY-NC 4.0",
        "size": "23,113 annotated sentences",
    },
}
