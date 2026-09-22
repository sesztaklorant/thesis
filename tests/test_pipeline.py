"""
test_pipeline.py — Gyors füstteszt (smoke test) szintetikus adaton.

Futtatás a projekt gyökeréből:
    python tests/test_pipeline.py

Mit csinál? Legyárt 600 kamu hotel-reviewt, és végigfuttatja rajtuk a
teljes pipeline-t: tisztítás -> címkézés -> felosztás -> TF-IDF + 3 modell
-> kiértékelés -> ABSA. Ha ez lefut, a környezeted rendben van, és
nyugodtan nekiállhatsz a valódi adatnak.

Ez nem a modell minőségét teszteli (a szintetikus adaton az eredmény
irreleváns), hanem azt, hogy a KÓD hibamentesen végigfut.
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
    """Szintetikus hotel reviewk, amelyekben van valódi jel (signal)."""
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
        if r < 0.5:            # pozitív
            text = " ".join(rng.choice(pos_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(8.2, 10.0)
        elif r < 0.8:          # negatív
            text = " ".join(rng.choice(neg_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(2.5, 5.8)
        else:                  # semleges
            text = " ".join(rng.choice(neu_bits, size=rng.integers(2, 4), replace=False))
            score = rng.uniform(6.2, 7.8)
        rows.append({"text": text, "score": round(float(score), 1)})

    df = pd.DataFrame(rows)
    df["review_id"] = [f"fake_{i}" for i in range(len(df))]
    df["dataset"] = "synthetic"
    df["score_scale_max"] = 10.0
    return df[["review_id", "text", "score", "score_scale_max", "dataset"]]


def main() -> int:
    ok = lambda msg: print(f"  [OK]   {msg}")
    print("=" * 66)
    print(" FÜSTTESZT — szintetikus adaton")
    print("=" * 66)

    set_seed()
    PATHS.ensure()
    ok(f"seed = {CFG.random_seed}, mappák rendben")

    # --- 1. Adat ---
    raw = make_fake_reviews(600)
    assert len(raw) == 600
    ok(f"szintetikus adat: {len(raw)} sor")

    # --- 2. Tisztítás + címkézés ---
    df = pp.build_dataset(raw, balance="none")
    assert {"text_clean", "text_bert", "sentiment"} <= set(df.columns)
    dist = df["sentiment"].value_counts()
    ok(f"előfeldolgozás kész: {len(df)} sor, eloszlás = {dist.to_dict()}")
    assert dist.min() > 0, "Valamelyik osztály üres lett!"

    # --- 3. Felosztás ---
    tr, va, te = pp.train_val_test_split(df)
    total = len(tr) + len(va) + len(te)
    assert total == len(df), "A felosztás sorokat veszített!"
    assert abs(len(tr) / total - 0.60) < 0.02, "A train arány nem 60%"
    ok(f"felosztás: {len(tr)} / {len(va)} / {len(te)} (60/20/20)")

    # a rétegzés tényleg működik?
    p_all = (df["sentiment"].astype(str) == "positive").mean()
    p_te = (te["sentiment"].astype(str) == "positive").mean()
    assert abs(p_all - p_te) < 0.03, "A rétegzett felosztás nem tartotta az arányt"
    ok("rétegzés (stratify) ellenőrizve")

    # --- 4. Klasszikus modellek ---
    summary, fitted = mc.train_and_evaluate_all(tr, va, save=False)
    assert len(summary) == 3
    print(summary[["accuracy", "f1_macro", "f1_weighted"]].round(3).to_string())
    assert summary["f1_macro"].max() > 0.5, "Gyanúsan rossz eredmény — nézd meg a pipeline-t"
    ok("Naive Bayes / LogReg / LinearSVM betanítva és kiértékelve")

    # --- 5. Kiértékelés ---
    best = summary.index[0]
    y_pred = fitted[best].predict(te["text_clean"])
    m = ev.compute_metrics(te["sentiment"].astype(str), y_pred)
    assert set(m) >= {"accuracy", "f1_macro", "f1_weighted"}
    ok(f"teszt metrikák ({best}): acc={m['accuracy']:.3f}, macro-F1={m['f1_macro']:.3f}")

    rep = ev.per_class_report(te["sentiment"].astype(str), y_pred)
    assert len(rep) >= 3
    ok("osztályonkénti riport elkészült")

    errs = ev.error_analysis(te, y_pred, text_col="text", n_examples=3)
    ok(f"hibaelemzés: {len(errs)} mintapélda kigyűjtve")

    # --- 6. Ábra mentése (headless backend) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev.plot_confusion_matrix(te["sentiment"].astype(str), y_pred,
                             title=f"Confusion matrix — {best}")
    p = ev.save_figure("_smoketest_confusion")
    assert p.exists()
    plt.close("all")
    ok(f"confusion matrix elmentve: {p.name}")

    # --- 7. Top jellemzők ---
    top = mc.top_features_per_class(fitted["logreg"], top_n=5)
    assert len(top) == 15
    ok("top jellemzők osztályonként: " + ", ".join(
        top[top.sentiment == "negative"]["jellemző"].head(5)))

    # --- 8. ABSA ---
    sent_df = asp.explode_to_sentences(df, text_col="text")
    assert len(sent_df) > len(df)
    sent_df = asp.add_aspect_columns(sent_df, text_col="sentence")
    ok(f"mondatra bontás: {len(sent_df)} mondat, "
       f"átlag {sent_df['n_aspects'].mean():.2f} aspektus/mondat")

    at = asp.aspect_sentiment_table(sent_df, min_count=5)
    assert len(at) > 0, "Egy aspektust sem talált — nézd meg a lexikont"
    print(at[["említések", "negative", "positive", "neg_lift"]].head(6).to_string())
    ok(f"aspektus×sentiment tábla: {len(at)} aspektus")

    ev_spans = asp.extract_evidence(
        "the room was extremely hot because there was no air conditioning", "room")
    assert ev_spans, "A bizonyíték-kiemelés nem talált semmit"
    ok(f'bizonyíték-kiemelés: "{ev_spans[0]}"')

    pairs = asp.build_absa_pairs(sent_df.head(50))
    assert {"aspect", "sentence"} <= set(pairs.columns)
    ok(f"ABSA sentence-pair formátum: {len(pairs)} pár")

    # --- 9. Takarítás ---
    p.unlink(missing_ok=True)

    print("=" * 66)
    print(" MINDEN TESZT SIKERES — a környezeted készen áll.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
