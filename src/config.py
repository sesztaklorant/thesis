"""
config.py — The CENTRAL settings of the project.

Why is this needed?
-------------------
The most important requirement of a thesis is REPRODUCIBILITY: if the
reviewer or the supervisor asks "which threshold did you use to form the
neutral class?", you can point to a single file. Every "magic number"
(random seed, split ratios, thresholds, paths) lives HERE and is not
hard-coded anywhere else.

Usage from any notebook or module:

    from src.config import CFG, PATHS
    print(CFG.random_seed)
    print(PATHS.raw)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# --------------------------------------------------------------------------
# 1) PATHS
# --------------------------------------------------------------------------
# Project root = the parent of this file's parent folder.
# This way it works whether it is run from a notebook or from a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    # data
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"          # untouched download
    interim: Path = PROJECT_ROOT / "data" / "interim"  # intermediate, cleaned
    processed: Path = PROJECT_ROOT / "data" / "processed"  # ready for modelling

    # results
    results: Path = PROJECT_ROOT / "results"
    figures: Path = PROJECT_ROOT / "results" / "figures"   # figures for the thesis
    tables: Path = PROJECT_ROOT / "results" / "tables"     # tables (csv)
    models: Path = PROJECT_ROOT / "results" / "models"     # saved models

    def ensure(self) -> None:
        """Create every folder if it does not exist yet."""
        for p in [
            self.data, self.raw, self.interim, self.processed,
            self.results, self.figures, self.tables, self.models,
        ]:
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# --------------------------------------------------------------------------
# 2) EXPERIMENT SETTINGS
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Reproducibility ---
    # Passed to every operation that uses randomness (split, model init,
    # shuffle). This is what makes a re-run give the SAME result.
    random_seed: int = 42

    # --- Data split (60/20/20 as in Milestone 1) ---
    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    # Stratified split: every subset has the same class proportions.
    # For an imbalanced 3-class task this is MANDATORY.
    stratify: bool = True

    # --- Sentiment labelling from the guest score ---
    # The Booking.com / 515K datasets have NO ready-made sentiment label,
    # only a 1–10 score. Three classes have to be derived from it.
    #
    #   score <= neg_threshold           -> negative
    #   neg_threshold < score < pos_thr  -> neutral
    #   score >= pos_threshold           -> positive
    #
    # IMPORTANT: this decision has to be JUSTIFIED in the thesis (see
    # MUNKATERV.md, phase 3). The current cut is 7/9. The 6/8 cut common in
    # the literature was tested first, but on this data it leaves only ~3% of
    # the reviews negative (28.8x imbalance between the largest and smallest
    # class). With 7/9 the split is ~13% / 22% / 64% (4.9x imbalance).
    # The comparison of the alternatives (6/8, 7/9, 7/10, 8/10) is in
    # notebook 01, section 4, and in results/tables/01_label_scheme_comparison.csv.
    score_scale_max: float = 10.0
    neg_threshold: float = 7.0
    pos_threshold: float = 9.0

    # Canonical order of the three classes. Used EVERYWHERE, so that the
    # axes of the confusion matrices are consistent.
    labels: tuple[str, ...] = ("negative", "neutral", "positive")

    # --- Sampling (during development) ---
    # We do not develop on 1.6 million rows. The pipeline first runs on a
    # small sample, and sampling is switched off only for the final run.
    dev_sample_size: int | None = 50_000   # None = full dataset
    # Class balancing: the neutral class is typically a minority.
    # "downsample" = the large classes are reduced to the size of the smallest.
    balance_strategy: str = "none"  # "none" | "downsample"

    # --- Text preparation ---
    min_chars: int = 10         # shorter text is treated as noise
    max_chars: int = 5_000      # cut off extremely long outliers
    drop_duplicates: bool = True
    keep_english_only: bool = True

    # --- TF-IDF (classical ML) ---
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)   # unigram + bigram
    tfidf_min_df: int = 3        # rarer words are dropped (noise filtering)
    tfidf_max_df: float = 0.9    # words occurring in >90% of documents are dropped
    tfidf_sublinear_tf: bool = True

    # --- Transformer (BERT) ---
    model_name: str = "bert-base-uncased"

    # max_length: how many tokens of the text the model looks at.
    #
    # Measured on val.csv: 95% of the reviews are shorter than 130 words,
    # 99% shorter than 215 words. In English 1 word ≈ 1.3 tokens, so the 95th
    # percentile is about 170 and the 99th about 280 tokens. The earlier
    # value of 128 tokens would have truncated well over 10% of the reviews.
    #
    # ⚠ THE DIRECTION OF TRUNCATION MATTERS: the text is built as
    #       title . positive_part   negative_part
    # so cutting off the end SYSTEMATICALLY loses the NEGATIVE content.
    # That would bias the model towards the positive class — hence the
    # generous max_length. Notebook 03 measures in a separate cell how many
    # reviews are affected by truncation and whether it skews the class
    # distribution.
    max_length: int = 256

    # batch_size: on the 4 GB RTX 3050 Ti a batch of 4 is safe with 256 tokens.
    # gradient_accumulation_steps=8 restores the EFFECTIVE batch of 32, so the
    # result is mathematically the same, it is just computed in more steps.
    batch_size: int = 32          # default for Colab T4 (16 GB)
    local_batch_size: int = 4     # own GPU (4 GB), with 256 tokens
    local_grad_accum: int = 8     # 4 × 8 = 32 effective batch
    learning_rate: float = 2e-5  # recommended in the BERT paper: 2e-5 / 3e-5 / 5e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # --- ABSA aspects (HRAST + keyword baseline) ---
    aspects: tuple[str, ...] = (
        "room", "staff", "cleanliness", "facilities", "location",
        "breakfast", "pool", "bathroom", "parking", "noise",
        "wifi", "value_for_money",
    )

    # --- Which datasets take part in the experiments? ---
    # As agreed with the supervisor, for now we work ONLY with the Booking.com
    # data. The loaders of the external datasets (TripAdvisor, 515K, HRAST)
    # are kept, but load_dataset_by_name() refuses to load them while this is
    # False. To include them later, just set it to True here.
    use_external_datasets: bool = False

    # --- Local data file (from Booking's official train/val/test split) ---
    # As agreed with the supervisor, we start with the ~200k-row val.csv.
    # IMPORTANT: this is Booking's OWN validation split, not our validation
    # set. See section 2 of notebook 01.
    local_csv_name: str = "val.csv"
    source_split_name: str = "booking_val_split"

    def to_json(self, path: Path) -> None:
        """Save the config next to the experiment — so for every result you
        will know EXACTLY which settings produced it."""
        d = asdict(self)
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


CFG = Config()


# --------------------------------------------------------------------------
# 3) COLOUR PALETTE — every figure uses the same one
# --------------------------------------------------------------------------
# Why is this here and not separately in every notebook?
# 1) Consistency: all figures of the thesis belong to one system.
# 2) COLOUR-VISION ACCESSIBILITY — this is the more important reason.
#
# "Red = bad, green = good" is the most common choice AND the worst possible
# one: red-green colour blindness (deuteranopia/protanopia) affects about 8%
# of men and merges exactly these two colours. In a printed thesis this is a
# real readability problem.
#
# Instead: a RED <-> BLUE diverging pair with a neutral grey midpoint.
# It expresses polarity (negative ... positive) and stays distinguishable
# with colour blindness. Validated: the difference between the two poles
# under simulated protanopia is ΔE 21.6 (well above the threshold of 8),
# with normal vision 32.3.
SENTIMENT_COLORS: dict[str, str] = {
    "negative": "#e34948",   # red
    "neutral":  "#898781",   # neutral grey (midpoint of the diverging scale)
    "positive": "#2a78d6",   # blue
}

# Single-hue sequential scale for magnitude (heatmap, frequency).
# From light to dark — NEVER a rainbow scale.
SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95",
]

# Diverging scale for "relative to the average" measures (e.g. neg_lift):
# blue = fewer negatives than average, red = more. Neutral in the middle.
DIVERGING_BLUE_RED: list[str] = [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a82c2b",
]

# Chart chrome: axes, grid lines, labels. Subdued, does not compete with
# the data.
INK = {
    "primary":   "#0b0b0b",
    "secondary": "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "axis":      "#c3c2b7",
    "surface":   "#ffffff",
}


def sentiment_palette(order: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """The three sentiment colours in CFG.labels order (or in the given order)."""
    return [SENTIMENT_COLORS[l] for l in (order or CFG.labels)]


def apply_plot_style() -> None:
    """
    Uniform matplotlib style for the whole thesis.
    Call it in the first cell of the notebooks.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,            # print quality
        "savefig.bbox": "tight",
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.spines.top": False,      # subdued frame
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
# 4) DATASET REGISTRY
# --------------------------------------------------------------------------
# One place that records where each dataset comes from and what its role is.
# The "Data Sources" chapter of the thesis can be filled in from this.
DATASETS = {
    "booking_hf": {
        "role": "main",  # main sentiment classification (RQ1, RQ2)
        "source": "huggingface",
        "id": "Booking-com/accommodation-reviews",
        "license": "CC BY-SA 4.0 (non-commercial)",
        "size": "~1.6M reviews, ~40k properties",
        "text_fields": ["review_title", "review_positive", "review_negative"],
        "score_field": "guest_score",
        "score_scale": (1.0, 10.0),
    },
    "kaggle_515k": {
        "role": "external_validation",  # RQ5 — cross-dataset generalisation
        "source": "kaggle",
        "id": "jiashenliu/515k-hotel-reviews-data-in-europe",
        "license": "Kaggle — terms given in the description (educational use)",
        "size": "~515k reviews, 1493 European hotels",
        "text_fields": ["Positive_Review", "Negative_Review"],
        "score_field": "Reviewer_Score",
        "score_scale": (2.5, 10.0),
    },
    "tripadvisor": {
        "role": "external_validation_alt",  # DIFFERENT platform -> stronger RQ5 test
        "source": "kaggle",
        "id": "andrewmvd/trip-advisor-hotel-reviews",
        "license": "CC BY-NC-SA 4.0",
        "size": "~20k reviews",
        "text_fields": ["Review"],
        "score_field": "Rating",
        "score_scale": (1.0, 5.0),
    },
    "hrast": {
        "role": "absa",  # RQ3, RQ4 — aspect-level annotation
        "source": "kaggle",
        "id": "costastziouvas/hotel-reviews-aspects-sentiments-and-topics",
        "license": "check on the Kaggle page and document it!",
        "size": "23,113 annotated sentences",
        "text_fields": ["sentence"],  # exact column name to be checked after download
        "score_field": None,
    },
}
