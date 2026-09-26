"""
evaluation.py — Metrikák, confusion matrix, hibaelemzés.

A dolgozat 8. fejezete ("Data Split and Evaluation").

Minden modellt (klasszikus ÉS BERT) UGYANEZEKKEL a függvényekkel értékelünk
ki — ez az egyetlen módja annak, hogy az RQ1 összehasonlítás korrekt legyen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import CFG, PATHS

log = logging.getLogger(__name__)


# ==========================================================================
# 1) METRIKÁK
# ==========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """
    A dolgozatban megígért metrikák, egy szótárban.

    Miért macro ÉS weighted F1?
      - weighted: a gyakori osztályok súlyozottan számítanak. Ha a pozitív
        osztály az adat 70%-a, egy "mindent pozitívnak mondok" modell is
        magas weighted értéket kap -> FÉLREVEZETŐ.
      - macro: minden osztály egyformán számít. Ez mutatja meg, hogy a
        modell tényleg megtanulta-e a nehéz neutral osztályt.

      A FŐ metrikád a MACRO-F1 legyen, és ezt írd is le a dolgozatban.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f_mac,
        "precision_weighted": p_wt,
        "recall_weighted": r_wt,
        "f1_weighted": f_wt,
    }


def baseline_metrics(y_true, labels: list[str] | None = None) -> pd.DataFrame:
    """
    TRIVIÁLIS ALAPMODELLEK — a számok, amiket minden modellednek vernie kell.

    Miért nélkülözhetetlen ez kiegyensúlyozatlan adatnál?
    ------------------------------------------------------
    Ha az adatod 64,5%-a pozitív, akkor egy modell, ami MINDENRE pozitívat
    mond, 64,5%-os pontosságot ér el anélkül, hogy bármit megtanult volna.
    Ha a dolgozatban 0,70-es accuracy szerepel kontextus nélkül, az
    lenyűgözően hangzik — pedig alig jobb a semminél.

    Ezért a dolgozat eredménytáblázatának ELSŐ SORA ez legyen. Minden további
    szám ehhez viszonyítva értelmezendő.

    Három alapmodell:
      • többségi osztály — mindig a leggyakoribbat tippeli
      • rétegzett véletlen — az osztályarányok szerint sorsol
      • egyenletes véletlen — 1/3–1/3–1/3
    """
    labels = labels or list(CFG.labels)
    y = np.asarray(y_true).astype(str)
    rng = np.random.default_rng(CFG.random_seed)

    props = pd.Series(y).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    majority = props.idxmax()

    rows = {}

    # 1) Mindig a többségi osztály
    rows[f"többségi osztály ('{majority}')"] = compute_metrics(
        y, np.full(len(y), majority))

    # 2) Rétegzett véletlen: az osztályarányok szerint sorsol
    rows["rétegzett véletlen"] = compute_metrics(
        y, rng.choice(labels, size=len(y), p=props.to_numpy()))

    # 3) Egyenletes véletlen
    rows["egyenletes véletlen"] = compute_metrics(
        y, rng.choice(labels, size=len(y)))

    out = pd.DataFrame(rows).T[["accuracy", "f1_macro", "f1_weighted"]].round(4)
    out.index.name = "alapmodell"

    log.info("A legerősebb triviális alapmodell: accuracy=%.4f, macro-F1=%.4f",
             out["accuracy"].max(), out["f1_macro"].max())
    return out


def class_weights(y_train, labels: list[str] | None = None) -> dict[str, float]:
    """
    Osztálysúlyok a veszteségfüggvényhez (főleg a BERT-hez).

    Képlet:  súly_c = n_összes / (osztályok_száma * n_c)

    Ez a scikit-learn `class_weight='balanced'` logikája. A ritkább osztály
    hibái arányosan nagyobb büntetést kapnak, így a modell nem éri be azzal,
    hogy a többséget tippeli.

    A klasszikus modelleknél ez már be van állítva (`class_weight='balanced'`
    a models_classical.py-ban). A transformernél KÉZZEL kell átadni a
    veszteségfüggvénynek — lásd a 03-as notebook WeightedTrainer osztályát.
    """
    labels = labels or list(CFG.labels)
    y = pd.Series(np.asarray(y_train).astype(str))
    counts = y.value_counts().reindex(labels, fill_value=0)
    n, k = len(y), (counts > 0).sum()
    weights = {c: (n / (k * m) if m > 0 else 0.0) for c, m in counts.items()}
    log.info("Osztálysúlyok: %s", {k2: round(v, 3) for k2, v in weights.items()})
    return weights


def per_class_report(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Osztályonkénti precision/recall/F1 táblázat — mehet a dolgozatba."""
    labels = labels or list(CFG.labels)
    rep = classification_report(
        np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str),
        labels=labels, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T


# ==========================================================================
# 2) CONFUSION MATRIX
# ==========================================================================
def plot_confusion_matrix(
    y_true, y_pred,
    labels: list[str] | None = None,
    title: str = "Confusion matrix",
    normalize: bool = True,
    save_as: str | None = None,
    ax=None,
):
    """
    Normalizált confusion matrix (soronként = az adott VALÓS osztály hány
    százalékát tippelte a modell melyik osztálynak).

    Erre figyelj a dolgozatban: 3 osztályos sentimentnél szinte mindig azt
    fogod látni, hogy a neutral osztály "szétfolyik" a másik kettőbe. Ez a
    legérdekesebb megfigyelésed lesz — és a hibaelemzés fejezet alapja.
    """
    import matplotlib.pyplot as plt

    labels = labels or list(CFG.labels)
    cm = confusion_matrix(
        np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=labels
    )
    cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.2, 4.4))

    im = ax.imshow(cm_show, cmap="Blues", vmin=0, vmax=cm_show.max())
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predikált osztály")
    ax.set_ylabel("Valós osztály")
    ax.set_title(title)

    thresh = cm_show.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{cm_show[i, j]:.1%}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if cm_show[i, j] > thresh else "black")

    if created:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        if save_as:
            save_figure(save_as)
    return ax


def save_figure(name: str, dpi: int = 300) -> Path:
    """
    Ábra mentése a results/figures/ mappába, 300 DPI-vel.

    300 DPI a nyomdai minőség — a szakdolgozatba beillesztve nem lesz
    pixeles. Mentsd el MINDEN ábrát, amit használni akarsz; sokkal
    kellemetlenebb három hónap múlva újrafuttatni a notebookot.
    """
    import matplotlib.pyplot as plt
    PATHS.ensure()
    path = PATHS.figures / (name if name.endswith((".png", ".pdf")) else f"{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Ábra elmentve: %s", path)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Táblázat mentése CSV-be, hogy a dolgozatba be tudd másolni."""
    PATHS.ensure()
    path = PATHS.tables / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, encoding="utf-8")
    log.info("Táblázat elmentve: %s", path)
    return path


# ==========================================================================
# 3) HIBAELEMZÉS
# ==========================================================================
def error_analysis(
    df: pd.DataFrame,
    y_pred,
    text_col: str = "text",
    label_col: str = "sentiment",
    n_examples: int = 10,
    seed: int = CFG.random_seed,
) -> pd.DataFrame:
    """
    Kigyűjti a félreosztályozott reviewkat, hibatípusonként csoportosítva.

    A dolgozat 8. fejezete kifejezetten kéri a "misclassified reviews"
    elemzését. Futtasd ezt, olvasd el kézzel 20-30 példát, és írj róluk
    néhány bekezdést — ez a fajta kvalitatív elemzés sokkal többet ér a
    bírálónál, mint még egy táblázat.
    """
    out = df.copy()
    out["predicted"] = np.asarray(y_pred).astype(str)
    out["true"] = out[label_col].astype(str)
    errors = out[out["predicted"] != out["true"]].copy()
    errors["hibatípus"] = errors["true"] + " -> " + errors["predicted"]

    log.info("Összes hiba: %d / %d (%.1f%%)",
             len(errors), len(out), 100 * len(errors) / max(len(out), 1))
    log.info("Leggyakoribb hibatípusok:\n%s",
             errors["hibatípus"].value_counts().to_string())

    samples = (errors.groupby("hibatípus", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), n_examples), random_state=seed)))
    cols = ["hibatípus", "true", "predicted", "score", text_col]
    return samples[[c for c in cols if c in samples.columns]].reset_index(drop=True)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Több modell metrikáit egy táblázatba rendezi — ez lesz a dolgozat
    fő összehasonlító táblázata (RQ1).

        compare_models({
            "Naive Bayes":  metrics_nb,
            "Logistic Reg": metrics_lr,
            "Linear SVM":   metrics_svm,
            "BERT":         metrics_bert,
        })
    """
    return (pd.DataFrame(results).T
            [["accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted"]]
            .sort_values("f1_macro", ascending=False)
            .round(4))
