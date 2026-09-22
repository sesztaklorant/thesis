"""
models_classical.py — TF-IDF + klasszikus gépi tanulás (RQ1, RQ2).

A dolgozat 6.2. fejezete.

Három modell, ahogy a Milestone 1-ben szerepel:
  1. Multinomial Naive Bayes — gyors, egyszerű, erős baseline szövegen
  2. Logistic Regression     — jól kalibrált valószínűségek, értelmezhető súlyok
  3. Linear SVM              — szöveges osztályozásban gyakran a legjobb lineáris modell

Mindhárom scikit-learn Pipeline-ba van csomagolva (TF-IDF + osztályozó),
mert így:
  - nem szivárog információ a teszthalmazból a TF-IDF illesztésébe
    (ez a leggyakoribb, és a bírálók által legszívesebben keresett hiba!)
  - a GridSearchCV egyben tudja hangolni a vektorizálót és a modellt
  - egyetlen .joblib fájlba menthető az egész
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import CFG, PATHS

log = logging.getLogger(__name__)


# ==========================================================================
# 1) TF-IDF vektorizáló
# ==========================================================================
def make_vectorizer(**overrides) -> TfidfVectorizer:
    """
    TF-IDF beállítások a config.py-ból, felülírható paraméterekkel.

    Mit jelentenek?
      ngram_range=(1,2) : egyszavas ÉS kétszavas kifejezések. A "not clean"
                          bigram sentiment feladatnál aranyat ér.
      min_df=3          : a 3-nál kevesebb dokumentumban előforduló szó zaj
                          (elgépelések, egyedi nevek) -> kidobjuk
      max_df=0.9        : a szinte minden reviewban szereplő szó ("hotel")
                          nem hordoz megkülönböztető információt
      sublinear_tf=True : 1 + log(tf) a nyers gyakoriság helyett. Tompítja
                          a sokszor ismételt szavak hatását.
    """
    params = dict(
        max_features=CFG.tfidf_max_features,
        ngram_range=CFG.tfidf_ngram_range,
        min_df=CFG.tfidf_min_df,
        max_df=CFG.tfidf_max_df,
        sublinear_tf=CFG.tfidf_sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


# ==========================================================================
# 2) Modellek
# ==========================================================================
def get_classifier(name: str, class_weight: str | None = "balanced"):
    """
    class_weight="balanced": a ritkább osztály (jellemzően a neutral)
    hibái nagyobb súlyt kapnak. Kiegyensúlyozatlan 3 osztályos feladatnál
    ez jelentősen javítja a macro-F1-et. A MultinomialNB nem támogatja,
    ott a class_prior marad az alapértelmezett.
    """
    name = name.lower()
    if name in {"nb", "naive_bayes", "multinomialnb"}:
        return MultinomialNB(alpha=0.5)
    if name in {"logreg", "logistic_regression", "lr"}:
        return LogisticRegression(
            max_iter=2000, C=1.0, class_weight=class_weight,
            random_state=CFG.random_seed,
        )
    if name in {"svm", "linear_svm", "linearsvc"}:
        return LinearSVC(
            C=1.0, class_weight=class_weight,
            random_state=CFG.random_seed, max_iter=5000,
        )
    raise ValueError(f"Ismeretlen modell: {name}")


def build_pipeline(model_name: str, vectorizer_kwargs: dict | None = None,
                   class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline([
        ("tfidf", make_vectorizer(**(vectorizer_kwargs or {}))),
        ("clf", get_classifier(model_name, class_weight=class_weight)),
    ])


MODEL_NAMES = ["nb", "logreg", "svm"]


# ==========================================================================
# 3) Tanítás és kiértékelés
# ==========================================================================
def train_and_evaluate_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "sentiment",
    model_names: list[str] | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Mindhárom modellt betanítja a tanító halmazon és a VALIDÁCIÓS halmazon
    értékeli. A teszthalmazhoz ilyenkor MÉG NEM NYÚLUNK — azt csak a
    végleges modellválasztás után, egyszer használjuk.

    Visszatér: (összefoglaló táblázat, {modellnév: betanított pipeline})
    """
    from .evaluation import compute_metrics

    model_names = model_names or MODEL_NAMES
    rows, fitted = [], {}

    X_tr, y_tr = train_df[text_col], train_df[label_col].astype(str)
    X_va, y_va = val_df[text_col], val_df[label_col].astype(str)

    for name in model_names:
        log.info("Tanítás: %s (%d minta)", name, len(X_tr))
        pipe = build_pipeline(name)
        pipe.fit(X_tr, y_tr)

        m = compute_metrics(y_va, pipe.predict(X_va))
        m["model"] = name
        m["n_train"] = len(X_tr)
        m["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
        rows.append(m)
        fitted[name] = pipe

        if save:
            save_model(pipe, f"tfidf_{name}")

    summary = (pd.DataFrame(rows)
               .set_index("model")
               .sort_values("f1_macro", ascending=False))
    return summary, fitted


def save_model(pipe: Pipeline, name: str) -> Path:
    import joblib
    PATHS.ensure()
    path = PATHS.models / f"{name}.joblib"
    joblib.dump(pipe, path)
    log.info("Modell elmentve: %s", path)
    return path


def load_model(name: str) -> Pipeline:
    import joblib
    return joblib.load(PATHS.models / f"{name}.joblib")


# ==========================================================================
# 4) ABLÁCIÓS VIZSGÁLAT — közvetlenül az RQ2-re válaszol
# ==========================================================================
def preprocessing_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "logreg",
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """
    RQ2: "Hogyan befolyásolja az előfeldolgozás és a jellemzőreprezentáció
    a klasszikus modellek teljesítményét?"

    Ez a függvény adja meg rá a SZÁMSZERŰ választ: ugyanazt a modellt
    futtatja különböző előfeldolgozási és TF-IDF beállításokkal, és
    táblázatot ad vissza. Ez a táblázat egy az egyben mehet a dolgozatba.

    Előfeltétel: a train_df / val_df tartalmazza a 'text' (nyers),
    'text_clean' (teljes tisztítás) oszlopokat.
    """
    from .evaluation import compute_metrics
    from .preprocessing import clean_text_classic

    variants: list[tuple[str, str, dict]] = [
        # (megnevezés, melyik szövegoszlop, TF-IDF felülírások)
        ("nyers szöveg, unigram",              "text",       {"ngram_range": (1, 1)}),
        ("nyers szöveg, uni+bigram",           "text",       {"ngram_range": (1, 2)}),
        ("teljes tisztítás, unigram",          "text_clean", {"ngram_range": (1, 1)}),
        ("teljes tisztítás, uni+bigram",       "text_clean", {"ngram_range": (1, 2)}),
        ("teljes tisztítás, uni+bi+trigram",   "text_clean", {"ngram_range": (1, 3)}),
        ("teljes tisztítás, sublinear_tf ki",  "text_clean", {"sublinear_tf": False}),
        ("teljes tisztítás, max_feat=10k",     "text_clean", {"max_features": 10_000}),
    ]

    # Extra változat: stopword-szűrés tagadószavak MEGTARTÁSA nélkül.
    # Ez gyakran mérhetően ront -> remek megfigyelés a dolgozatba.
    for split_df in (train_df, val_df):
        if "text_nonegation" not in split_df.columns:
            split_df["text_nonegation"] = split_df["text"].map(
                lambda t: clean_text_classic(t, keep_negations=False)
            )
    variants.append(("tisztítás, tagadószavak eldobva", "text_nonegation", {}))

    rows = []
    for label, col, kw in variants:
        pipe = build_pipeline(model_name, vectorizer_kwargs=kw)
        pipe.fit(train_df[col], train_df[label_col].astype(str))
        m = compute_metrics(val_df[label_col].astype(str), pipe.predict(val_df[col]))
        m["variáns"] = label
        m["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
        rows.append(m)
        log.info("%-38s macro-F1 = %.4f", label, m["f1_macro"])

    return (pd.DataFrame(rows)
            .set_index("variáns")
            .sort_values("f1_macro", ascending=False))


# ==========================================================================
# 5) ÉRTELMEZHETŐSÉG — melyik szavak döntenek?
# ==========================================================================
def top_features_per_class(pipe: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Kiszedi a lineáris modell legnagyobb súlyú jellemzőit osztályonként.

    Ez közvetlenül segít az RQ3-nál ("mely aspektusok kötődnek a negatív
    értékeléshez?") és remek, könnyen értelmezhető táblázatot ad a
    dolgozatba: látni fogod, hogy pl. a "dirty", "noisy", "smell" szavak
    hajtják a negatív osztályt.

    Csak LogisticRegression / LinearSVC esetén működik (NB-nél a
    log-valószínűségek mások, arra külön kezelés kell).
    """
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("Ez a modell nem lineáris (nincs coef_ attribútuma).")

    feature_names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "sentiment": cls,
                "rang": rank,
                "jellemző": feature_names[idx],
                "súly": float(coefs[idx]),
            })
    return pd.DataFrame(rows)
