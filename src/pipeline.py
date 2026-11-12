"""Command-line entry point. Each step starts from the saved output of the previous one.

    python -m src.pipeline prepare       # load and clean the Booking data
    python -m src.pipeline label-a       # (A) zero-shot labelling
    python -m src.pipeline train-b       # (B) fine-tuning on HRAST
    python -m src.pipeline label-b       # (B) labelling
    python -m src.pipeline train-c       # (C) self-training
    python -m src.pipeline label-c       # (C) labelling
    python -m src.pipeline combine       # final label
    python -m src.pipeline export-gold   # manual annotation sample
    python -m src.pipeline all           # steps prepare ... combine in sequence
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from . import set_seed
from .config import CFG, PATHS

log = logging.getLogger(__name__)

PREPARED = PATHS.processed / "booking_prepared"
WORKING_SET = PATHS.processed / "booking_working_set"
LABELLED = PATHS.processed / "booking_labelled"


# ==========================================================================
# STEPS
# ==========================================================================
def step_prepare() -> pd.DataFrame:
    """Load, filter, clean, select the working set."""
    from .data_loading import load_booking, write_frame
    from .evaluation import save_table
    from .preprocessing import build_dataset

    prepared, filter_log = build_dataset(load_booking())
    write_frame(prepared, PREPARED)
    save_table(filter_log.set_index("step"), "01_filtering_log")

    # All three approaches run on the same working set, otherwise they are not comparable.
    n = CFG.labelling_sample_size
    working = prepared.sample(n, random_state=CFG.random_seed) if n and n < len(prepared) else prepared
    write_frame(working.reset_index(drop=True), WORKING_SET)
    CFG.to_json(PATHS.processed / "config_snapshot.json")
    log.info("Prepared: %d rows | working set: %d rows", len(prepared), len(working))
    return working


def _working_set() -> pd.DataFrame:
    from .data_loading import read_frame
    return read_frame(WORKING_SET)


def step_label_a() -> pd.DataFrame:
    from .bert_labelling import label_approach_a, save_labels
    labels = label_approach_a(_working_set())
    save_labels(labels, "labels_A")
    return labels


def step_train_b() -> dict:
    from .bert_training import train_approach_b
    from .data_loading import load_hrast_raw, prepare_hrast_sentences
    from .evaluation import save_table

    result = train_approach_b(prepare_hrast_sentences(load_hrast_raw()))
    save_table(pd.DataFrame([result["test_metrics"]], index=["HRAST test"]), "02_approach_b_hrast_test")
    return result


def step_label_b() -> pd.DataFrame:
    from .bert_labelling import label_approach_b, save_labels
    labels = label_approach_b(_working_set())
    save_labels(labels, "labels_B")
    return labels


def _load_available_labels(approaches=("A", "B", "C")) -> pd.DataFrame:
    """Join the available label tables on review_id."""
    from .bert_labelling import load_labels
    from .data_loading import frame_exists

    frames = [load_labels(f"labels_{a}") for a in approaches
              if frame_exists(PATHS.labels / f"labels_{a}")]
    if not frames:
        raise FileNotFoundError("No label table found in data/labels/ — run label-a first.")
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame, how="inner")
    return joined


def step_train_c() -> dict:
    from .bert_labelling import select_pseudo_labels
    from .bert_training import train_approach_c

    teachers = _load_available_labels(("A", "B"))
    pseudo = select_pseudo_labels(teachers)
    return train_approach_c(_working_set(), pseudo)


def step_label_c() -> pd.DataFrame:
    from .bert_labelling import label_approach_c, save_labels
    labels = label_approach_c(_working_set())
    save_labels(labels, "labels_C")
    return labels


def step_combine() -> pd.DataFrame:
    """Final label + join with the reviews."""
    from .bert_labelling import combine_approaches, save_labels
    from .data_loading import write_frame

    combined = combine_approaches(_load_available_labels(), approaches=CFG.final_label_approaches)
    save_labels(combined, "labels_final")
    labelled = _working_set().set_index("review_id").join(combined, how="inner").reset_index()
    write_frame(labelled, LABELLED)
    log.info("Final label distribution: %s",
             labelled["label_final"].value_counts(normalize=True).round(3).to_dict())
    return labelled


def step_export_gold():
    from .label_validation import export_gold_sample
    return export_gold_sample(_working_set())


STEPS = {
    "prepare": step_prepare, "label-a": step_label_a, "train-b": step_train_b,
    "label-b": step_label_b, "train-c": step_train_c, "label-c": step_label_c,
    "combine": step_combine, "export-gold": step_export_gold,
}


def run_all() -> None:
    """
    Steps 1–7 in sequence. If HRAST is unavailable, (B) is skipped and (C)
    learns from (A) only - the process does not stop.
    """
    step_prepare()
    step_label_a()
    try:
        step_train_b()
        step_label_b()
    except FileNotFoundError as e:
        log.warning("Approach B skipped: %s", e)
    step_train_c()
    step_label_c()
    step_combine()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BERT-based sentiment labelling of Booking.com reviews")
    parser.add_argument("step", choices=[*STEPS, "all"])
    args = parser.parse_args(argv)
    set_seed()
    PATHS.ensure()
    if args.step == "all":
        run_all()
    else:
        result = STEPS[args.step]()
        if isinstance(result, dict):
            log.info("Result: %s", json.dumps({k: str(v)[:200] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
