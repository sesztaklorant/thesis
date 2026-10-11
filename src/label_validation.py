"""
label_validation.py — A BERT-címkék ellenőrzése valódi címke HIÁNYÁBAN.
                      Validating the BERT labels in the ABSENCE of a true label.

HU: A gépi címke csak akkor használható, ha meg tudjuk mutatni, hogy JÓ. Mivel
    nincs kész "igazság", öt, egymást kiegészítő bizonyítékot gyűjtünk:

      1. Egyezés a megközelítések között   — Cohen- és Fleiss-kappa
      2. Kapcsolat a vendégpontszámmal     — Spearman-korreláció (konvergens validitás:
                                              erős, de NEM tökéletes kapcsolat várható)
      3. Booking-mezőpróba                 — a "pozitív" mező szövegét pozitívnak, a
                                              "negatív" mezőét negatívnak kell mondania
                                              (a vendég maga döntötte el, hova ír)
      4. Kézi gold minta                   — kb. 400 vélemény VAKON annotálva; ez az
                                              egyetlen közvetlen pontosságmérés
      5. Pontszám-vágás referencia         — megmutatja, mennyire térne el a naiv vágás

EN: A machine label is only usable if we can show that it is GOOD. Since there
    is no ready "truth", we collect five complementary pieces of evidence:

      1. Agreement between the approaches  — Cohen's and Fleiss' kappa
      2. Relation to the guest score       — Spearman correlation (convergent validity:
                                              strong but NOT perfect relation expected)
      3. Booking field test                — text from the "positive" field should be
                                              called positive, from the "negative" field
                                              negative (the guest chose the field)
      4. Manual gold sample                — ~400 reviews annotated BLIND; the only
                                              direct accuracy measurement
      5. Score-cut reference               — shows how far the naive cut would deviate
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import APPROACH_COLORS, CFG, INK, PATHS, SENTIMENT_COLORS

log = logging.getLogger(__name__)

LABELS: list[str] = list(CFG.labels)
ORDINAL = {label: i for i, label in enumerate(LABELS)}   # HU/EN: negative < neutral < positive


def available_approaches(labels: pd.DataFrame,
                         candidates: tuple[str, ...] = ("A", "B", "C")) -> list[str]:
    return [a for a in candidates if f"{a}_label" in labels.columns]


def comparison_mask(labels: pd.DataFrame) -> pd.Series:
    """
    HU: Az összevetésben használható sorok: kizárjuk azokat, amelyeken a (C) tanult.
    EN: Rows usable in the comparison: excludes the reviews (C) was trained on.
    """
    if "C_in_training" in labels.columns:
        return ~labels["C_in_training"].fillna(False).astype(bool)
    return pd.Series(True, index=labels.index)


# ==========================================================================
# 5) PONTSZÁM-VÁGÁS — CSAK REFERENCIA / SCORE CUT — REFERENCE ONLY
# ==========================================================================
def score_cut_labels(scores: pd.Series, neg_threshold: float = CFG.reference_neg_threshold,
                     pos_threshold: float = CFG.reference_pos_threshold) -> pd.Series:
    """
    HU: A korábbi, NEM érvényes címkézés (score ≤ 7 negatív, ≥ 9 pozitív), kizárólag
        összehasonlításra: hány véleménynél mond mást a szöveg, mint a pontszám-vágás?
    EN: The earlier, NOT valid labelling (score ≤ 7 negative, ≥ 9 positive), only
        for comparison: for how many reviews does the text say something else?
    """
    s = pd.to_numeric(scores, errors="coerce")
    out = pd.Series(np.where(s <= neg_threshold, "negative",
                             np.where(s >= pos_threshold, "positive", "neutral")),
                    index=scores.index)
    return out.where(s.notna())


# ==========================================================================
# 1) ELOSZLÁS ÉS EGYEZÉS / DISTRIBUTION AND AGREEMENT
# ==========================================================================
def label_distribution(labels: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    HU: Osztályarányok címkeforrásonként. columns: {megjelenítendő név: oszlop}.
    EN: Class shares per label source. columns: {display name: column}.
    """
    rows = {name: labels[col].astype(str).value_counts(normalize=True).reindex(LABELS, fill_value=0)
            for name, col in columns.items() if col in labels.columns}
    out = pd.DataFrame(rows).T
    out["n"] = [int(labels[col].notna().sum()) for name, col in columns.items()
                if col in labels.columns]
    return out.round(4)


def cohen_kappa(a: pd.Series, b: pd.Series, weights: str | None = None) -> float:
    """
    HU: Cohen-kappa: véletlen egyezésre korrigált egyezés. weights="linear" esetén a
        sorrendet is figyelembe veszi (negatív↔pozitív tévesztés nagyobb hiba, mint
        negatív↔semleges). Értelmezés (Landis & Koch, 1977): 0,41–0,60 közepes,
        0,61–0,80 jelentős, 0,81 felett szinte tökéletes.
    EN: Cohen's kappa: agreement corrected for chance. With weights="linear" it
        respects the order (negative↔positive is a bigger error than
        negative↔neutral). Interpretation (Landis & Koch, 1977): 0.41–0.60
        moderate, 0.61–0.80 substantial, above 0.81 almost perfect.
    """
    from sklearn.metrics import cohen_kappa_score
    both = pd.concat([a, b], axis=1).dropna()
    if both.empty:
        return float("nan")
    return float(cohen_kappa_score(both.iloc[:, 0].map(ORDINAL), both.iloc[:, 1].map(ORDINAL),
                                   labels=list(range(len(LABELS))), weights=weights))


def pairwise_agreement(labels: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    HU: Minden címkeforrás-pár: egyezési arány, Cohen-kappa, lineárisan súlyozott kappa.
    EN: Every pair of label sources: agreement rate, Cohen's kappa, linearly weighted kappa.
    """
    rows = []
    for (name_a, col_a), (name_b, col_b) in itertools.combinations(columns.items(), 2):
        if col_a not in labels.columns or col_b not in labels.columns:
            continue
        pair = labels[[col_a, col_b]].dropna()
        rows.append({
            "pair": f"{name_a} vs {name_b}",
            "n": len(pair),
            "agreement": float((pair[col_a].astype(str) == pair[col_b].astype(str)).mean()),
            "cohen_kappa": cohen_kappa(pair[col_a].astype(str), pair[col_b].astype(str)),
            "weighted_kappa": cohen_kappa(pair[col_a].astype(str), pair[col_b].astype(str), "linear"),
        })
    return pd.DataFrame(rows).set_index("pair").round(4)


def fleiss_kappa(labels: pd.DataFrame, columns: list[str]) -> float:
    """
    HU: Fleiss-kappa: kettőnél több "annotátor" (itt: megközelítés) együttes egyezése.
    EN: Fleiss' kappa: joint agreement of more than two "raters" (here: approaches).
    """
    data = labels[columns].dropna().astype(str)
    if data.empty or len(columns) < 2:
        return float("nan")
    counts = np.stack([(data == label).sum(axis=1).to_numpy() for label in LABELS], axis=1)
    n_items, n_raters = counts.shape[0], len(columns)
    p_item = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()
    p_class = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_class ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def approach_crosstab(labels: pd.DataFrame, col_a: str, col_b: str, normalize: bool = True) -> pd.DataFrame:
    """HU: Két címkeforrás kereszttáblája. / EN: Crosstab of two label sources."""
    tab = pd.crosstab(labels[col_a].astype(str), labels[col_b].astype(str),
                      normalize="index" if normalize else False)
    return tab.reindex(index=LABELS, columns=LABELS, fill_value=0).round(4)


# ==========================================================================
# 2) KAPCSOLAT A VENDÉGPONTSZÁMMAL / RELATION TO THE GUEST SCORE
# ==========================================================================
def guest_score_validity(data: pd.DataFrame, approaches: list[str],
                         score_col: str = "score") -> pd.DataFrame:
    """
    HU: Megközelítésenként:
          spearman_rho      — a szövegalapú pontszám és a vendégpontszám rangkorrelációja
          mean_guest_score_<címke> — átlagos vendégpontszám a címke szerint
          neg_but_score_ge9 — "negatív" szöveg, de ≥ 9-es pontszám (szöveg ≠ szám)
          pos_but_score_le6 — "pozitív" szöveg, de ≤ 6-os pontszám
        Erős, de nem tökéletes korreláció a JÓ eredmény: ha tökéletes lenne, a
        szövegalapú címke semmit sem adna a pontszámhoz képest.
    EN: Per approach:
          spearman_rho      — rank correlation of the text-based score and the guest score
          mean_guest_score_<label> — mean guest score per label
          neg_but_score_ge9 — "negative" text but a score ≥ 9 (text ≠ number)
          pos_but_score_le6 — "positive" text but a score ≤ 6
        A strong but imperfect correlation is the GOOD outcome: if it were
        perfect, the text-based label would add nothing over the score.
    """
    from scipy.stats import spearmanr

    rows = []
    guest = pd.to_numeric(data[score_col], errors="coerce")
    for a in approaches:
        score_col_a = f"{a}_score_cont" if f"{a}_score_cont" in data.columns else None
        row = {"approach": a}
        if score_col_a:
            valid = data[score_col_a].notna() & guest.notna()
            row["spearman_rho"] = float(spearmanr(data.loc[valid, score_col_a], guest[valid]).statistic)
        label = data[f"{a}_label"].astype(str)
        for lab in LABELS:
            row[f"mean_guest_score_{lab}"] = float(guest[label == lab].mean())
        row["neg_but_score_ge9"] = float(((label == "negative") & (guest >= 9)).mean())
        row["pos_but_score_le6"] = float(((label == "positive") & (guest <= 6)).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("approach").round(4)


def text_score_discrepancies(data: pd.DataFrame, score_col_text: str = "score10_final",
                             score_col_guest: str = "score", min_gap: int = 4,
                             n_examples: int = 20, text_col: str = "text",
                             seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Példák, ahol a szöveg és a pontszám nagyon eltér (|eltérés| ≥ min_gap).
        Kézzel elolvasva ezek mutatják meg, MIÉRT nem lehet a pontszámot vágni
        (pl. 10 pont, de a szöveg panaszkodik) — kvalitatív anyag a dolgozatba.
    EN: Examples where text and score differ strongly (|gap| ≥ min_gap). Read by
        hand, these show WHY the score cannot be cut (e.g. 10 points but the text
        complains) — qualitative material for the thesis.
    """
    gap = data[score_col_text] - pd.to_numeric(data[score_col_guest], errors="coerce")
    picked = data.assign(gap=gap)[gap.abs() >= min_gap]
    log.info("Reviews with |text score − guest score| ≥ %d: %d (%.2f%%)",
             min_gap, len(picked), 100 * len(picked) / max(len(data), 1))
    cols = [c for c in ["gap", score_col_guest, score_col_text, "label_final", text_col]
            if c in picked.columns]
    return picked.sample(min(n_examples, len(picked)), random_state=seed)[cols]


# ==========================================================================
# 3) BOOKING-MEZŐPRÓBA / BOOKING FIELD TEST
# ==========================================================================
def build_field_test_set(data: pd.DataFrame, n_per_field: int = 2_000, min_words: int = 4,
                         seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Ismert polaritású szövegek a Booking-szerkezetből: a review_positive mező
        szövege "positive", a review_negative mezőé "negative". (Semleges itt nincs —
        ez a próba a két szélső osztályt ellenőrzi.)
    EN: Texts with known polarity from the Booking structure: text of the
        review_positive field is "positive", of review_negative "negative".
        (No neutral here — this test checks the two extreme classes.)
    """
    from .preprocessing import clean_text_minimal

    parts = []
    for field, expected in (("review_positive", "positive"), ("review_negative", "negative")):
        texts = data[field].fillna("").astype(str)
        pool = data.loc[texts.str.split().str.len() >= min_words, ["review_id", field]]
        sample = pool.sample(min(n_per_field, len(pool)), random_state=seed)
        parts.append(pd.DataFrame({
            "review_id": sample["review_id"].astype(str) + f"__{expected[:3]}",
            "text_bert": sample[field].map(clean_text_minimal),
            "expected": expected,
        }))
    return pd.concat(parts, ignore_index=True)


def field_test_report(expected: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """
    HU: Mezőnként: a várt osztályt hány százalékban találta el, hány százalékot
        mondott semlegesnek, és hányat az ELLENKEZŐJÉNEK (ez a súlyos hiba).
    EN: Per field: share of the expected class hit, share called neutral, and the
        share called the OPPOSITE (the serious error).
    """
    frame = pd.DataFrame({"expected": expected.to_numpy(), "predicted": predicted.astype(str).to_numpy()})
    rows = []
    for exp, group in frame.groupby("expected"):
        opposite = "negative" if exp == "positive" else "positive"
        rows.append({"field": f"review_{exp}", "n": len(group),
                     "hit": (group["predicted"] == exp).mean(),
                     "neutral": (group["predicted"] == "neutral").mean(),
                     "opposite": (group["predicted"] == opposite).mean()})
    return pd.DataFrame(rows).set_index("field").round(4)


# ==========================================================================
# 4) KÉZI GOLD MINTA / MANUAL GOLD SAMPLE
# ==========================================================================
def export_gold_sample(data: pd.DataFrame, n: int = CFG.gold_sample_size, text_col: str = "text",
                       seed: int = CFG.random_seed, filename: str = "gold_sample_to_annotate.csv") -> Path:
    """
    HU: Véletlen minta kézi annotálásra. VAK annotálás: a fájlban csak a szöveg
        van — sem a vendégpontszám, sem a modellek címkéje nem látszik, különben
        az annotátort befolyásolná. A kitöltendő oszlopok:
          gold_label  — negative / neutral / positive
          gold_score  — 1–10 egész (opcionális)
        Egyszerű véletlen minta -> a mért pontosság torzítatlan becslés.
    EN: Random sample for manual annotation. BLIND annotation: the file contains
        only the text — neither the guest score nor the model labels are shown,
        otherwise they would bias the annotator. Columns to fill in:
          gold_label  — negative / neutral / positive
          gold_score  — integer 1–10 (optional)
        Simple random sample -> the measured accuracy is an unbiased estimate.
    """
    PATHS.ensure()
    sample = data.sample(min(n, len(data)), random_state=seed)
    out = pd.DataFrame({"review_id": sample["review_id"], "text": sample[text_col],
                        "gold_label": "", "gold_score": ""})
    path = PATHS.annotation / filename
    if path.exists():
        raise FileExistsError(f"{path} already exists — it may contain your annotations. "
                              "Rename or delete it first.")
    out.to_csv(path, index=False, encoding="utf-8-sig")   # HU/EN: utf-8-sig -> Excel opens it correctly
    log.info("Gold sample exported: %s (%d reviews). Fill in gold_label (and gold_score).", path, len(out))
    return path


def gold_csv_to_xlsx(csv_name: str = "gold_sample_to_annotate.csv",
                     xlsx_name: str = "gold_sample_to_annotate.xlsx") -> Path:
    """
    HU: A gold minta CSV -> formázott Excel a kényelmes annotáláshoz:
          • külön oszlopok (a magyar Excel a vesszős CSV-t egy oszlopba tenné);
          • a szöveg tördelve, széles oszlopban; az első sor rögzítve;
          • gold_label: legördülő lista (negative / neutral / positive);
          • gold_score: csak 1–10 közötti egész fogadható el.
        A load_gold() ezután az .xlsx-et olvassa (ha létezik).
    EN: Gold sample CSV -> formatted Excel for convenient annotation:
          • separate columns (Hungarian Excel would put a comma CSV into one column);
          • text wrapped in a wide column; first row frozen;
          • gold_label: drop-down list (negative / neutral / positive);
          • gold_score: only integers 1–10 are accepted.
        load_gold() then reads the .xlsx (if it exists).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    xlsx_path = PATHS.annotation / xlsx_name
    if xlsx_path.exists():
        raise FileExistsError(f"{xlsx_path} already exists — it may contain your annotations.")
    data = pd.read_csv(PATHS.annotation / csv_name, encoding="utf-8-sig")

    wb = Workbook()
    ws = wb.active
    ws.title = "annotation"
    columns = ["review_id", "text", "gold_label", "gold_score"]
    ws.append(columns)
    for row in data[columns].itertuples(index=False):
        ws.append([None if pd.isna(v) or v == "" else v for v in row])

    n = len(data) + 1
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col, width in zip("ABCD", (26, 110, 14, 12)):
        ws.column_dimensions[col].width = width
    for (cell,) in ws.iter_rows(min_row=2, max_row=n, min_col=2, max_col=2):
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    label_list = DataValidation(type="list", formula1='"negative,neutral,positive"', allow_blank=True)
    score_range = DataValidation(type="whole", operator="between", formula1="1", formula2="10",
                                 allow_blank=True)
    ws.add_data_validation(label_list)
    ws.add_data_validation(score_range)
    label_list.add(f"C2:C{n}")
    score_range.add(f"D2:D{n}")

    wb.save(xlsx_path)
    log.info("Excel for annotation: %s (%d reviews)", xlsx_path, len(data))
    return xlsx_path


def load_gold(filename: str | None = None) -> pd.DataFrame:
    """
    HU: A kitöltött gold minta beolvasása és ellenőrzése. Alapból az .xlsx-et olvassa,
        ha az létezik, különben a .csv-t.
    EN: Read and check the annotated gold sample. By default it reads the .xlsx if it
        exists, otherwise the .csv.
    """
    if filename is None:
        xlsx = PATHS.annotation / "gold_sample_to_annotate.xlsx"
        filename = xlsx.name if xlsx.exists() else "gold_sample_to_annotate.csv"
    path = PATHS.annotation / filename
    gold = (pd.read_excel(path) if path.suffix == ".xlsx"
            else pd.read_csv(path, encoding="utf-8-sig"))
    log.info("Gold sample read from: %s", path.name)
    gold["gold_label"] = gold["gold_label"].astype(str).str.strip().str.lower()
    valid = gold["gold_label"].isin(LABELS)
    if (~valid).any():
        log.warning("%d rows without a valid gold_label are ignored.", int((~valid).sum()))
    gold = gold[valid].copy()
    gold["gold_score"] = pd.to_numeric(gold.get("gold_score"), errors="coerce")
    return gold.set_index("review_id")


def evaluate_on_gold(gold: pd.DataFrame, labels: pd.DataFrame, columns: dict[str, str],
                     score_columns: dict[str, str] | None = None) -> pd.DataFrame:
    """
    HU: Minden címkeforrás összevetése a kézi címkével: accuracy, makro-F1, kappa,
        és (ha van gold_score) a pontszám átlagos abszolút hibája (MAE).
        Ide érdemes a pontszám-vágás referenciát is betenni: ez méri SZÁMSZERŰEN,
        mennyivel jobb a szövegalapú címke a vágásnál.
    EN: Compare every label source with the manual label: accuracy, macro-F1,
        kappa and (if gold_score exists) the mean absolute error of the score.
        Include the score-cut reference too: it measures QUANTITATIVELY how much
        better the text-based label is than the cut.
    """
    from .evaluation import compute_metrics

    joined = gold.join(labels, how="inner", rsuffix="_lbl")
    log.info("Gold reviews matched with labels: %d", len(joined))
    rows = []
    for name, col in columns.items():
        if col not in joined.columns:
            continue
        m = compute_metrics(joined["gold_label"], joined[col].astype(str))
        row = {"source": name, "n": len(joined), "accuracy": m["accuracy"],
               "f1_macro": m["f1_macro"],
               "cohen_kappa": cohen_kappa(joined["gold_label"], joined[col].astype(str))}
        score_col = (score_columns or {}).get(name)
        if score_col and score_col in joined.columns and joined["gold_score"].notna().any():
            ok = joined["gold_score"].notna()
            row["score_mae"] = float((joined.loc[ok, score_col] - joined.loc[ok, "gold_score"]).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("source").sort_values("f1_macro", ascending=False).round(4)


def calibrate_neutral_band(gold: pd.DataFrame, labels: pd.DataFrame, prefix: str = "B",
                           bands: tuple[float, ...] = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
                           ) -> pd.DataFrame:
    """
    HU: A semleges sáv kalibrálása a gold mintán: minden sávszélességre kiszámolja a
        kézi címkével mért accuracy-t, makro-F1-et, kappát és a semleges osztály F1-ét.
        Figyelem: ugyanazon a mintán választunk és mérünk -> az eredmény enyhén
        optimista; egyetlen paraméternél ez elfogadható, de a dolgozatban jelezni kell.
    EN: Calibrating the neutral band on the gold sample: for every band width it computes
        accuracy, macro-F1, kappa and the F1 of the neutral class against the manual label.
        Note: we choose and measure on the same sample -> slightly optimistic; acceptable
        for a single parameter, but it has to be stated in the thesis.
    """
    from sklearn.metrics import f1_score

    from .bert_labelling import apply_neutral_band
    from .evaluation import compute_metrics

    joined = gold[["gold_label"]].join(labels, how="inner")
    rows = []
    for band in bands:
        banded = apply_neutral_band(joined, prefix, band)
        pred = banded[f"{prefix}_label"].astype(str)
        m = compute_metrics(joined["gold_label"], pred)
        rows.append({"band": band, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                     "cohen_kappa": cohen_kappa(joined["gold_label"], pred),
                     "f1_neutral": f1_score(joined["gold_label"], pred, labels=["neutral"],
                                            average="macro", zero_division=0),
                     "share_neutral": float((pred == "neutral").mean())})
    return pd.DataFrame(rows).set_index("band").round(4)


def bootstrap_ci(gold: pd.DataFrame, labels: pd.DataFrame, columns: dict[str, str],
                 n_boot: int = 2000, seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: 95%-os bootstrap konfidencia-intervallum a makro-F1-re és az accuracy-re.
        400 véleménynél a különbségek egy része a véletlen ingadozáson belül lehet —
        ez mutatja meg, melyik eltérés valódi.
    EN: 95% bootstrap confidence interval for macro-F1 and accuracy. With 400 reviews
        some differences may be within random variation — this shows which are real.
    """
    from .evaluation import compute_metrics

    joined = gold[["gold_label"]].join(labels, how="inner")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(joined))
    rows = []
    for name, col in columns.items():
        if col not in joined.columns:
            continue
        y_true, y_pred = joined["gold_label"].to_numpy(), joined[col].astype(str).to_numpy()
        f1s, accs = [], []
        for _ in range(n_boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            m = compute_metrics(y_true[b], y_pred[b])
            f1s.append(m["f1_macro"])
            accs.append(m["accuracy"])
        rows.append({"source": name,
                     "f1_macro_low": np.percentile(f1s, 2.5), "f1_macro_high": np.percentile(f1s, 97.5),
                     "accuracy_low": np.percentile(accs, 2.5), "accuracy_high": np.percentile(accs, 97.5)})
    return pd.DataFrame(rows).set_index("source").round(4)


# ==========================================================================
# 6) ÁBRÁK / FIGURES
# ==========================================================================
def plot_label_distribution(dist: pd.DataFrame, title: str = "Label distribution by source",
                            save_as: str | None = None):
    """HU: 100%-ra halmozott vízszintes oszlopok forrásonként. / EN: 100% stacked horizontal bars per source."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    data = dist[LABELS]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(data) + 1.8))
    left = np.zeros(len(data))
    for label in LABELS:
        ax.barh(data.index, data[label], left=left, color=SENTIMENT_COLORS[label], label=label,
                height=0.7, edgecolor=INK["surface"], linewidth=1.5)
        for y, (x0, w) in enumerate(zip(left, data[label])):
            if w > 0.07:
                ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += data[label].to_numpy()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def plot_score_vs_guest(data: pd.DataFrame, approaches: list[str], score_col: str = "score",
                        save_as: str | None = None):
    """
    HU: Átlagos szövegalapú pontszám a vendégpontszám függvényében, megközelítésenként.
        Az átló (y = x) a tökéletes egyezés; az eltérés a "szöveg ≠ szám" jelenség.
    EN: Mean text-based score as a function of the guest score, per approach.
        The diagonal (y = x) is perfect agreement; the gap is the "text ≠ number" effect.
    """
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    guest = pd.to_numeric(data[score_col], errors="coerce").round()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([1, 10], [1, 10], ls="--", color=INK["axis"], lw=1.2, label="y = x")
    for a in approaches:
        col = f"{a}_score_cont"
        if col not in data.columns:
            continue
        means = data.groupby(guest)[col].mean()
        ax.plot(means.index, means.values, "o-", color=APPROACH_COLORS.get(a), label=a, lw=2)
    ax.set(xlabel="Guest score (rounded)", ylabel="Mean text-based score (1–10)",
           xlim=(0.5, 10.5), ylim=(0.5, 10.5))
    ax.set_title("Text-based score vs guest score", loc="left")
    ax.legend()
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def plot_agreement_matrix(tab: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                          save_as: str | None = None):
    """HU: Soronként normalizált kereszttábla hőtérképként. / EN: Row-normalised crosstab as a heatmap."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tab.columns)), tab.columns, rotation=20)
    ax.set_yticks(range(len(tab.index)), tab.index)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.grid(visible=False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else INK["primary"])
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def paired_gold_test(gold: pd.DataFrame, pred_a: pd.Series, pred_b: pd.Series,
                     names: tuple[str, str] = ("A", "B"), n_boot: int = 2000,
                     seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Két címkéző PÁROS összevetése ugyanazon a gold mintán: McNemar-próba (csak az egyik helyes
        esetek aránya) és bootstrap 95%-os konfidenciaintervallum a különbségre (2. − 1.) pontosságban,
        macro-F1-ben és kappában. Ha az intervallum tartalmazza a 0-t, a különbség nem szignifikáns.
    EN: PAIRED comparison of two labellers on the same gold sample: McNemar test (cases where only one
        is right) and a bootstrap 95% confidence interval for the difference (2nd − 1st) in accuracy,
        macro-F1 and kappa. If the interval contains 0, the difference is not significant.
    """
    from scipy.stats import binomtest
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    y = gold["gold_label"]
    a = pd.Series(pred_a).reindex(y.index)
    b = pd.Series(pred_b).reindex(y.index)
    ok = a.notna() & b.notna()
    y, a, b = y[ok].to_numpy(), a[ok].to_numpy(), b[ok].to_numpy()
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    p_mcnemar = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0

    metrics = {"accuracy": accuracy_score,
               "f1_macro": lambda t, p: f1_score(t, p, average="macro", labels=list(LABELS), zero_division=0),
               "cohen_kappa": cohen_kappa_score}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rows = []
    for name, fn in metrics.items():
        diffs = []
        for _ in range(n_boot):
            s = rng.choice(idx, len(idx))
            diffs.append(fn(y[s], b[s]) - fn(y[s], a[s]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"metric": name, f"{names[0]}": fn(y, a), f"{names[1]}": fn(y, b),
                     f"diff ({names[1]} − {names[0]})": fn(y, b) - fn(y, a), "ci_low": lo, "ci_high": hi,
                     "significant": not (lo <= 0 <= hi)})
    table = pd.DataFrame(rows).set_index("metric").round(4)
    table.attrs.update({"only_a_right": only_a, "only_b_right": only_b, "mcnemar_p": float(p_mcnemar)})
    log.info("Paired gold test: only %s right %d | only %s right %d | McNemar p = %.3f",
             names[0], only_a, names[1], only_b, p_mcnemar)
    return table
