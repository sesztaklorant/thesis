"""Smoke test on synthetic data.

600 generated reviews go through cleaning, labelling, splitting, TF-IDF with three
classifiers, evaluation and the aspect analysis. The test checks that the code runs,
not the quality of the results.

    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import set_seed
from src.config import CFG, PATHS
from src import preprocessing as pp
from src import models_classical as mc
from src import evaluation as ev
from src import aspects as asp


def make_fake_reviews(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic hotel reviews that contain a real signal."""
    rng = np.random.default_rng(seed)

    pos_bits = [
        "The room was spacious and spotless.",
        "The staff were incredibly friendly and helpful.",
        "Great location, right next to the metro station.",
        "Breakfast was excellent with a huge buffet.",
        "Very good value for money.",
        "The bed was comfortable and the shower had great water pressure.",
    ]
    neg_bits = [
        "The room was extremely hot because there was no air conditioning.",
        "The bathroom was dirty and smelled bad.",
        "The walls were thin and the street noise was unbearable.",
        "The wifi connection did not work at all.",
        "Reception staff were rude and unhelpful.",
        "Completely overpriced for what you get.",
    ]
    neu_bits = [
        "The hotel was okay, nothing special.",
        "The room was average in size.",
        "Breakfast was standard continental.",
        "Parking was available but you had to pay extra.",
        "It was fine for a one night stay.",
    ]

    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5:  # positive
            text = " ".join(rng.choice(pos_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(9.0, 10.0)
        elif r < 0.8:  # negative
            text = " ".join(rng.choice(neg_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(2.5, 6.8)
        else:  # neutral
            text = " ".join(rng.choice(neu_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(7.2, 8.8)
        rows.append({"text": text, "score": round(float(score), 1)})

    df = pd.DataFrame(rows)
    df["review_id"] = [f"fake_{i}" for i in range(len(df))]
    df["dataset"] = "synthetic"
    df["score_scale_max"] = 10.0
    return df[["review_id", "text", "score", "score_scale_max", "dataset"]]


def main() -> int:
    ok = lambda msg: print(f"  [OK]   {msg}")
    print("=" * 66)
    print(" SMOKE TEST — on synthetic data")
    print("=" * 66)

    set_seed()
    PATHS.ensure()
    ok(f"seed = {CFG.random_seed}, folders in place")

    # --- 1. Data ---
    raw = make_fake_reviews(600)
    assert len(raw) == 600
    ok(f"synthetic data: {len(raw)} rows")

    # --- 2. Cleaning + labelling ---
    df = pp.build_dataset(raw, balance="none")
    assert {"text_clean", "text_bert", "sentiment"} <= set(df.columns)
    dist = df["sentiment"].value_counts()
    ok(f"preprocessing done: {len(df)} rows, distribution = {dist.to_dict()}")
    assert dist.min() > 0, "One of the classes became empty!"

    # --- 3. Split ---
    tr, va, te = pp.train_val_test_split(df)
    total = len(tr) + len(va) + len(te)
    assert total == len(df), "The split lost rows!"
    assert abs(len(tr) / total - 0.60) < 0.02, "The train share is not 60%"
    ok(f"split: {len(tr)} / {len(va)} / {len(te)} (60/20/20)")

    # stratification keeps the class shares
    p_all = (df["sentiment"].astype(str) == "positive").mean()
    p_te = (te["sentiment"].astype(str) == "positive").mean()
    assert abs(p_all - p_te) < 0.03, "The stratified split did not keep the proportions"
    ok("stratification checked")

    # --- 4. Classical models ---
    summary, fitted = mc.train_and_evaluate_all(tr, va, save=False)
    assert len(summary) == 3
    print(summary[["accuracy", "f1_macro", "f1_weighted"]].round(3).to_string())
    assert summary["f1_macro"].max() > 0.5, "Suspiciously bad result — check the pipeline"
    ok("Naive Bayes / LogReg / LinearSVM trained and evaluated")

    # --- 5. Evaluation ---
    best = summary.index[0]
    y_pred = fitted[best].predict(te["text_clean"])
    m = ev.compute_metrics(te["sentiment"].astype(str), y_pred)
    assert set(m) >= {"accuracy", "f1_macro", "f1_weighted"}
    ok(f"test metrics ({best}): acc={m['accuracy']:.3f}, macro-F1={m['f1_macro']:.3f}")

    rep = ev.per_class_report(te["sentiment"].astype(str), y_pred)
    assert len(rep) >= 3
    ok("per-class report done")

    errs = ev.error_analysis(te, y_pred, text_col="text", n_examples=3)
    ok(f"error analysis: {len(errs)} examples collected")

    # On the synthetic data the model is flawless, so the call above runs on
    # an empty table - which hid the pandas 3.0 column loss. So it is also
    # run deliberately with bad predictions, and the columns are checked.
    rng_err = np.random.default_rng(0)
    bad_pred = rng_err.choice(list(CFG.labels), size=len(te))
    errs_bad = ev.error_analysis(te, bad_pred, text_col="text", n_examples=3)
    assert len(errs_bad) > 0, "The error analysis is empty with deliberate errors"
    for col in ("error_type", "true", "predicted", "text"):
        assert col in errs_bad.columns, f"Missing column in the error analysis: {col}"
    assert errs_bad.groupby("error_type").size().max() <= 3, "n_examples is not respected"
    ok(f"error analysis with deliberate errors: {errs_bad['error_type'].nunique()} error types, "
       f"columns fine")

    bal = pp.balance_classes(df, "downsample")
    assert "sentiment" in bal.columns, "Balancing lost the sentiment column"
    assert bal["sentiment"].value_counts().nunique() == 1, "The class sizes are not equal"
    ok(f"balancing: {len(bal)} rows, {len(bal) // 3} per class")

    # --- 5/b. Baselines and class weights ---
    base = ev.baseline_metrics(te["sentiment"])
    assert len(base) == 3 and base.index.name == "baseline"
    ok(f"trivial baselines: best macro-F1 = {base['f1_macro'].max():.3f}")

    w = ev.class_weights(tr["sentiment"])
    assert list(w) == list(CFG.labels)
    counts = tr["sentiment"].astype(str).value_counts()
    assert abs(sum(w[c] * counts[c] for c in CFG.labels) - len(tr)) < 1e-6, \
        "The class weights do not follow the 'balanced' formula"
    ok("class weights: " + ", ".join(f"{c}={v:.2f}" for c, v in w.items()))

    # --- 6. Save a figure (headless backend) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev.plot_confusion_matrix(te["sentiment"].astype(str), y_pred,
                             title=f"Confusion matrix — {best}")
    p = ev.save_figure("_smoketest_confusion")
    assert p.exists()
    plt.close("all")
    ok(f"confusion matrix saved: {p.name}")

    # --- 7. Top features ---
    top = mc.top_features_per_class(fitted["logreg"], top_n=5)
    assert len(top) == 15
    ok("top features per class: " + ", ".join(
        top[top.sentiment == "negative"]["feature"].head(5)))

    # --- 8. ABSA ---
    sent_df = asp.explode_to_sentences(df, text_col="text")
    assert len(sent_df) > len(df)
    sent_df = asp.add_aspect_columns(sent_df, text_col="sentence")
    ok(f"sentence splitting: {len(sent_df)} sentences, "
       f"mean {sent_df['n_aspects'].mean():.2f} aspects/sentence")

    at = asp.aspect_sentiment_table(sent_df, min_count=5)
    assert len(at) > 0, "No aspect found — check the lexicon"
    print(at[["mentions", "negative", "positive", "neg_lift"]].head(6).to_string())
    ok(f"aspect×sentiment table: {len(at)} aspects")

    ev_spans = asp.extract_evidence(
        "the room was extremely hot because there was no air conditioning", "room")
    assert ev_spans, "Evidence extraction found nothing"
    ok(f'evidence extraction: "{ev_spans[0]}"')

    pairs = asp.build_absa_pairs(sent_df.head(50))
    assert {"aspect", "sentence"} <= set(pairs.columns)
    ok(f"ABSA sentence-pair format: {len(pairs)} pairs")

    # --- 9. Cleanup ---
    p.unlink(missing_ok=True)

    print("=" * 66)
    print(" ALL TESTS PASSED — your environment is ready.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
