"""
test_offline.py — Gyors tesztek GPU, internet és torch NÉLKÜL.
                  Quick tests WITHOUT GPU, internet and torch.

HU: A BERT-modellt egy kulcsszavas "ál-modell" (FakeScorer) helyettesíti, így a
    teljes logika (címkeképzés, 1–10 pontszám, aggregálás, pseudo-címkék, végső
    címke, validáció) ellenőrizhető másodpercek alatt.
EN: The BERT model is replaced by a keyword "fake model" (FakeScorer), so the
    whole logic (labels, 1–10 score, aggregation, pseudo-labels, final label,
    validation) can be checked in seconds.

    python -m pytest tests/          # HU/EN: if pytest is installed
    python tests/test_offline.py     # HU/EN: without pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bert_labelling as bl          # noqa: E402
from src import label_validation as lv        # noqa: E402
from src import preprocessing as pp           # noqa: E402
from src.data_loading import prepare_hrast_sentences, assemble_review_text  # noqa: E402


# --------------------------------------------------------------------------
# HU: Segédek / EN: Helpers
# --------------------------------------------------------------------------
class FakeScorer:
    """HU: Kulcsszavas ál-modell a TransformerScorer helyett. / EN: Keyword fake model instead of TransformerScorer."""

    def __init__(self, kind: str = "3class"):
        self.kind = kind

    def predict_proba(self, texts):
        out = []
        for t in texts:
            t = str(t).lower()
            pos = sum(w in t for w in ("great", "perfect", "lovely", "excellent"))
            neg = sum(w in t for w in ("dirty", "rude", "noisy", "terrible"))
            if self.kind == "stars":
                p = np.full(5, 0.05)
                p[4 if pos > neg else 0 if neg > pos else 2] = 0.8
            else:
                p = np.array([0.1, 0.8, 0.1])
                if pos > neg:
                    p = np.array([0.05, 0.1, 0.85])
                elif neg > pos:
                    p = np.array([0.85, 0.1, 0.05])
            out.append(p / p.sum())
        return np.array(out)


def fake_reviews(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pos_texts = ["Great location and lovely staff.", "Perfect breakfast!", "Excellent room."]
    neg_texts = ["The room was dirty.", "Rude reception staff.", "Very noisy street, terrible night."]
    rows = []
    for i in range(n):
        p = pos_texts[i % 3] if i % 4 != 0 else ""
        q = neg_texts[i % 3] if i % 3 == 0 else ""
        rows.append({"review_id": f"r{i}", "review_title": "Stay", "review_positive": p,
                     "review_negative": q, "score": float(rng.integers(5, 11))})
    df = pd.DataFrame(rows)
    df["text"] = assemble_review_text(df["review_title"], df["review_positive"], df["review_negative"])
    df["text_bert"] = df["text"].map(pp.clean_text_minimal)
    return df


# --------------------------------------------------------------------------
# HU: Tesztek / EN: Tests
# --------------------------------------------------------------------------
def test_score_mappings():
    _, cont, integer = bl.stars_to_score(np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
                                                   [0, 0, 1, 0, 0], [0, 0, 0, .5, .5]], float))
    assert list(integer) == [1, 10, 6, 9]          # 3★ -> 5.5 -> 6 (half up); 4.5★ -> 8.875 -> 9
    assert np.isclose(cont[2], 5.5)
    _, integer = bl.polarity_to_score(np.array([-1.0, 0.0, 1.0]))
    assert list(integer) == [1, 6, 10]
    assert list(bl.round_half_up([4.5, 5.5, 2.49])) == [5, 6, 2]


def test_parse_model_labels():
    assert bl.parse_model_labels({0: "negative", 1: "neutral", 2: "positive"})[0] == "3class"
    kind, order = bl.parse_model_labels({0: "Positive", 1: "Negative", 2: "Neutral"})
    assert kind == "3class" and order == ["positive", "negative", "neutral"]
    kind, stars = bl.parse_model_labels({i: f"{i + 1} star{'s' if i else ''}" for i in range(5)})
    assert kind == "stars" and stars == [1, 2, 3, 4, 5]


def test_sentence_split_and_aggregation():
    assert pp.split_sentences("Great room. Dirty bathroom!") == ["Great room.", "Dirty bathroom!"]
    assert pp.split_sentences("ok") == ["ok"]                 # HU/EN: never loses a review
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.1, 0.8, 0.1]])
    mean, index, extra = bl.aggregate_sentences(probs, pd.Series(["a", "a", "b"]))
    assert list(index) == ["a", "b"]
    assert bool(extra.loc["a", "is_mixed"]) and not bool(extra.loc["b", "is_mixed"])
    assert np.allclose(mean[0], [0.475, 0.05, 0.475])


def test_full_offline_flow():
    df = fake_reviews()
    a = bl.label_approach_a(df, use_cache=False,
                            scorers={"3class": FakeScorer(), "stars": FakeScorer("stars")})
    b = bl.label_with_classifier(df, "fake", "B", granularity="sentence", scorer=FakeScorer())
    assert {"A_label", "A_score10", "A_score10_polarity", "B_label", "B_is_mixed"} <= set(a.columns) | set(b.columns)
    assert a["A_score10"].between(1, 10).all()

    labels = a.join(b)
    pseudo = bl.select_pseudo_labels(labels, min_confidence=0.7, max_per_class=5)
    assert pseudo["pseudo_label"].value_counts().max() <= 5
    c = bl.label_with_classifier(df, "fake", "C", scorer=FakeScorer())
    final = bl.combine_approaches(labels.join(c))
    assert final["label_final"].isin(["negative", "neutral", "positive"]).all()
    assert final["score10_final"].between(1, 10).all()

    columns = {"A": "A_label", "B": "B_label", "C": "C_label"}
    agree = lv.pairwise_agreement(final, columns)
    assert agree["agreement"].between(0, 1).all()
    assert -1 <= lv.fleiss_kappa(final, list(columns.values())) <= 1

    data = df.set_index("review_id").join(final)
    validity = lv.guest_score_validity(data, ["A", "B", "C"])
    assert "spearman_rho" in validity.columns

    field = lv.build_field_test_set(df, n_per_field=10, min_words=2)
    report = lv.field_test_report(field["expected"],
                                  pd.Series(np.array(bl.LABELS)[FakeScorer().predict_proba(field["text_bert"]).argmax(1)]))
    assert report.loc["review_positive", "hit"] > 0.9


def test_hrast_preparation():
    raw = pd.DataFrame({"Sentence": ["Room was great.", "Room was great.", "Bad wifi but nice pool.",
                                     "Bad wifi but nice pool.", "It is a hotel."],
                        "Polarity": ["Positive", "Positive", "Negative", "Positive", "Neutral"],
                        "Aspect": ["room", "view", "wifi", "pool", "none"]})
    out = prepare_hrast_sentences(raw)
    assert len(out) == 2                                     # HU/EN: the conflicting sentence is dropped
    assert set(out["label"]) == {"positive", "neutral"}


def test_hrast_onehot_format():
    # HU: A Kaggle-HRAST valódi szerkezete (széles, 0/1 szentiment-oszlopok).
    # EN: The real structure of the Kaggle HRAST (wide, 0/1 sentiment columns).
    raw = pd.DataFrame({"id": [1, 2, 3, 4, 5], "review": ["1 elevator not working.", "Great pool.",
                        "Nice room but noisy.", "We arrived at noon.", "Great pool."],
                        "Unnamed: 2": np.nan, "positive": [0, 1, 1, 0, 1], "negative": [1, 0, 1, 0, 0],
                        "neutral": [0, 0, 0, 1, 0], "Lift": [1, 0, 0, 0, 0], "Aspect": 0})
    out = prepare_hrast_sentences(raw)
    assert dict(zip(out["sentence"], out["label"])) == {
        "1 elevator not working.": "negative", "Great pool.": "positive", "We arrived at noon.": "neutral"}


def test_neutral_band():
    probs = np.array([[0.5, 0.0, 0.5], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    frame = bl.probs_to_frame(probs, ["a", "b", "c", "d"], "B")
    assert list(frame["B_label"]) == ["negative", "positive", "negative", "neutral"]  # HU/EN: argmax (tie -> first)
    banded = bl.apply_neutral_band(frame, "B", band=0.3)
    assert list(banded["B_label"]) == ["neutral", "positive", "negative", "neutral"]
    assert banded["B_confidence"].between(0, 1).all()
    assert "B_label_argmax" in banded.columns
    table = bl.neutral_band_table(frame, "B")
    assert np.allclose(table.sum(axis=1), 1.0)


def test_score_cut_reference():
    cut = lv.score_cut_labels(pd.Series([3.0, 7.0, 8.0, 9.0, np.nan]))
    assert list(cut[:4]) == ["negative", "negative", "neutral", "positive"] and pd.isna(cut[4])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
