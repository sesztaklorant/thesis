"""Offline tests for the pure functions of feedback_checks (no GPU, internet or PyTorch).

    python tests/test_feedback_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bert_labelling as bl  # noqa: E402
from src import feedback_checks as fc  # noqa: E402


def test_band_labels_match_pipeline():
    """Nested CV uses the pipeline's band rule."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=500)
    frame = pd.DataFrame(probs, columns=[f"B_{s}" for s in bl.PROB_SUFFIXES])
    frame["B_label"] = "x"
    for band in (0.0, 0.1, 0.25, 0.5):
        expected = bl.apply_neutral_band(frame, "B", band=band)["B_label"].to_numpy()
        got = fc._band_labels(bl.polarity_from_probs(probs), band)
        assert (expected == got).all(), band


def test_wilson_interval():
    low, high = fc._wilson(50, 100)
    assert abs(low - 0.4038) < 1e-3 and abs(high - 0.5962) < 1e-3
    low, high = fc._wilson(0, 20)
    assert abs(low) < 1e-12 and 0 < high < 0.2


def test_complaint_units_flags():
    units = pd.DataFrame({
        "review_id": ["a", "a", "b", "c"],
        "field_polarity": ["negative", "negative", "negative", "positive"],
        "sentence": ["Nothing, everything was great", "The shower was cold", "Breakfast was fine", "Great staff"],
        "B_label": ["negative", "negative", "positive", "positive"],
    })
    out = fc._complaint_units(units)
    assert len(out) == 3  # praise field dropped
    assert out["non_complaint"].tolist() == [True, False, False]
    assert out["counted"].tolist() == [False, True, False]  # real and read as negative


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
