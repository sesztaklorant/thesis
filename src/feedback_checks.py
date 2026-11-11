"""
feedback_checks.py — A konzulensi visszajelzéshez kért ellenőrzések. / Checks requested in the supervisor's feedback.

HU: Nem új modell és nem új futtatás: a MEGLÉVŐ kimenetek (labels_B, field_units_B,
    subtopic_assignments_*, aranyminta, kézi témaellenőrzés) mélyebb kiértékelése.
      1) gold      — aranyminta: konfúziós mátrix, osztályonkénti mutatók, a semleges sáv hatása
                     osztályonként, a tévesztett negatív vélemények jellemzői
      2) band      — a semleges sáv beágyazott (nested) keresztvalidációja: a sávot a minta 4/5-én
                     választjuk, a maradék 1/5-én mérjük (a sávválasztás és a mérés elválasztva)
      3) high      — a 9–10 pontos vélemények panaszai a "nothing to complain about" típusú szövegek
                     nélkül, a modell által is negatívnak olvasott tagmondattal, szegmensenként, példákkal
      4) coverage  — a panaszmező tagmondatai: mennyit számol, és mennyit NEM a panaszelemzés
      5) subtopics — altémák a TELJES adaton: darabszám és arány aspektusonként, példamondatokkal,
                     a kézi ellenőrzés aspektusonként
      6) maps      — panasztérképek a kulcsmegállapításokhoz (fürdőszoba, zaj, ár-érték, személyzet,
                     recepció, szoba) — ugyanazzal a z-próbával és BH-korrekcióval, mint a 05-ös notebook
      7) export    — OPCIONÁLIS kézi ellenőrző fájlok (9–10 pontos panaszok; fürdőszoba- és zaj-altémák);
                     ha kitöltötted őket, ugyanez a parancs ki is értékeli
EN: No new model and no new run: a deeper evaluation of the EXISTING outputs (labels_B,
    field_units_B, subtopic_assignments_*, gold sample, manual topic check).
      1) gold      — gold sample: confusion matrix, per-class metrics, effect of the neutral band
                     per class, characteristics of the missed negative reviews
      2) band      — nested cross-validation of the neutral band: the band is chosen on 4/5 of the
                     sample and measured on the remaining 1/5 (selection and evaluation separated)
      3) high      — complaints in 9–10 reviews without "nothing to complain about"-type texts, with a
                     clause the model also reads as negative, by segment, with examples
      4) coverage  — complaint-field clauses: how many the complaint analysis counts and how many NOT
      5) subtopics — subtopics on the FULL data: count and share per aspect, example sentences,
                     the manual check per aspect
      6) maps      — complaint maps for the key findings (bathroom, noise, value for money, staff,
                     reception, room) — with the same z-test and BH correction as notebook 05
      7) export    — OPTIONAL manual check files (9–10 complaints; bathroom and noise subtopics);
                     once filled in, the same command also evaluates them

    python -m src.feedback_checks                 # minden ellenőrzés / every check (1–2 min, CPU)
    python -m src.feedback_checks gold band       # csak a kiválasztottak / only the selected ones

Kimenet / Output: results/tables/06_*.csv; az exportnál / for export: data/annotation/
                 high_score_complaint_check.xlsx, topic_check_negative_full2_bathroom_noise.xlsx
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .config import CFG, PATHS

log = logging.getLogger(__name__)

LABELS: list[str] = list(CFG.labels)
BAND_GRID: list[float] = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
HIGH_SCORE_MIN: int = 9            # HU: "magas pontszám" = 9–10 / EN: "high score" = 9–10
MIN_COMPLAINT_WORDS: int = 3       # HU: ugyanaz, mint a 01-es notebookban / EN: same as notebook 01
KEY_ASPECTS: tuple[str, ...] = ("bathroom", "noise", "room", "staff", "value_for_money",
                                "reception", "location", "cleanliness")
# HU: a dicséretoldalon a leggyakrabban dicsért aspektusok / EN: the most praised aspects on the praise side
POSITIVE_KEY_ASPECTS: tuple[str, ...] = ("location", "staff", "room", "cleanliness", "breakfast")
SEGMENTS: tuple[str, ...] = ("accommodation_type", "accommodation_country",
                             "accommodation_star_rating", "guest_type")


# ==========================================================================
# 0) BETÖLTÉS / LOADING
# ==========================================================================
def _gold_with_labels() -> pd.DataFrame:
    """HU: Aranyminta + (B) címkék + polaritás. / EN: Gold sample + (B) labels + polarity."""
    from .bert_labelling import PROB_SUFFIXES, load_labels, polarity_from_probs
    from .label_validation import load_gold

    data = load_gold().join(load_labels("labels_B"), how="inner")
    data["polarity"] = polarity_from_probs(data[[f"B_{s}" for s in PROB_SUFFIXES]].to_numpy())
    return data


def _band_labels(polarity, band: float) -> np.ndarray:
    """HU: Ugyanaz a szabály, mint az apply_neutral_band-ben. / EN: Same rule as in apply_neutral_band."""
    polarity = np.asarray(polarity, dtype=float)
    return np.where(np.abs(polarity) < band, "neutral", np.where(polarity > 0, "positive", "negative"))


def _complaint_units(units: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    HU: A panaszmező tagmondatai, "valódi panasz" jelzővel (nem "nothing"-típusú) és azzal, hogy a
        panaszelemzés számolja-e (valódi ÉS a (B) negatívnak olvassa — ugyanaz a kettős szűrés,
        mint a 05-ös notebookban).
    EN: Complaint-field clauses with a "real complaint" flag (not a "nothing"-type formula) and
        whether the complaint analysis counts it (real AND read as negative by (B) — the same
        double filter as in notebook 05).
    """
    from .data_loading import read_frame
    from .preprocessing import non_complaint_mask

    if units is None:
        units = read_frame(PATHS.labels / "field_units_B")
    out = units[units["field_polarity"] == "negative"].copy()
    out["non_complaint"] = non_complaint_mask(out["sentence"]).to_numpy()
    out["counted"] = ~out["non_complaint"] & (out["B_label"] == "negative")
    return out


def _review_meta() -> pd.DataFrame:
    """HU: Nyers vélemények (cache) + csillag 'n/a' kategória. / EN: Raw reviews (cache) + star 'n/a' category."""
    from .data_loading import load_booking
    from .descriptives import prepare_star_rating

    return prepare_star_rating(load_booking(verbose=False)).set_index("review_id")


# ==========================================================================
# 1) ARANYMINTA / GOLD SAMPLE
# ==========================================================================
def gold_tables(units: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    HU: Konfúziós mátrix (sor = kézi címke, oszlop = végső címke), osztályonkénti
        precision/recall/F1, a sáv hatása osztályonként, és a kézzel negatívnak ítélt
        vélemények jellemzői (vegyes-e, hol a polaritás, bekerül-e panaszuk a panaszelemzésbe).
    EN: Confusion matrix (row = manual label, column = final label), per-class
        precision/recall/F1, the effect of the band per class, and the characteristics of the
        reviews annotated as negative (mixed or not, where the polarity lies, whether their
        complaint enters the complaint analysis).
    """
    from .evaluation import per_class_report

    data = _gold_with_labels()
    confusion = (pd.crosstab(data["gold_label"], data["B_label"])
                 .reindex(index=LABELS, columns=LABELS, fill_value=0))
    confusion.index.name, confusion.columns.name = "gold (manual)", "predicted (final label)"
    per_class = per_class_report(data["gold_label"], data["B_label"]).round(4)

    rows = []
    for band in BAND_GRID:
        pred = _band_labels(data["polarity"], band)
        rep = per_class_report(data["gold_label"], pred)
        not_pos = ((pred != "positive") & (data["gold_label"] == "negative")).sum()
        rows.append({"band": band, "f1_macro": rep.loc["macro avg", "f1-score"],
                     **{f"recall_{c}": rep.loc[c, "recall"] for c in LABELS},
                     **{f"precision_{c}": rep.loc[c, "precision"] for c in LABELS},
                     "negatives_not_called_positive": not_pos / (data["gold_label"] == "negative").sum()})
    tradeoff = pd.DataFrame(rows).set_index("band").round(4)

    # HU: A kézzel negatív vélemények: tagmondatszintű kép a field_units_B-ből.
    # EN: Reviews annotated as negative: clause-level picture from field_units_B.
    complaints = _complaint_units(units)
    neg = data[data["gold_label"] == "negative"].copy()
    unit_view = (complaints[complaints["review_id"].isin(neg.index)]
                 .groupby("review_id")
                 .agg(complaint_clauses=("sentence", "size"), counted_complaint_clauses=("counted", "sum")))
    neg = neg.join(unit_view)
    neg[["complaint_clauses", "counted_complaint_clauses"]] = (
        neg[["complaint_clauses", "counted_complaint_clauses"]].fillna(0).astype(int))
    detail = neg[["B_label", "B_label_argmax", "polarity", "B_is_mixed", "B_n_sentences", "gold_score",
                  "complaint_clauses", "counted_complaint_clauses"]].sort_values(["B_label", "polarity"])
    summary = (neg.assign(argmax_negative=neg["B_label_argmax"] == "negative",
                          has_counted_complaint=neg["counted_complaint_clauses"] > 0,
                          abs_polarity_below_010=neg["polarity"].abs() < 0.10)
               .groupby("B_label")
               .agg(n=("B_label", "size"), mixed=("B_is_mixed", "sum"),
                    argmax_negative=("argmax_negative", "sum"),
                    mean_polarity=("polarity", "mean"), abs_polarity_below_010=("abs_polarity_below_010", "sum"),
                    mean_gold_score=("gold_score", "mean"),
                    has_counted_complaint=("has_counted_complaint", "sum"))
               .reindex(LABELS).round(3))
    summary.index.name = "gold negative, predicted as"
    log.info("Gold negatives with a counted complaint clause: %d of %d",
             int(summary["has_counted_complaint"].sum()), len(neg))
    return {"06_gold_confusion_matrix": confusion, "06_gold_per_class": per_class,
            "06_gold_band_tradeoff": tradeoff, "06_gold_negatives_detail": detail,
            "06_gold_negatives_summary": summary}


# ==========================================================================
# 2) A SEMLEGES SÁV BEÁGYAZOTT KERESZTVALIDÁCIÓJA / NESTED CV OF THE NEUTRAL BAND
# ==========================================================================
def band_nested_cv(n_splits: int = 5, n_repeats: int = 20, grid: list[float] | None = None,
                   seed: int = CFG.random_seed) -> dict[str, pd.DataFrame]:
    """
    HU: Ismételt rétegzett k-szoros felosztás: minden felosztásban a sávot a TANÍTÓ részeken
        választjuk (max. makro-F1 a rácson), és a KIHAGYOTT részen alkalmazzuk. Ismétlésenként
        mind a 399 vélemény pontosan egyszer kap "kihagyott" címkét; ezekből számoljuk a
        mutatókat, majd átlagolunk az ismétlésekre. Így a mért teljesítményt nem befolyásolja,
        hogy a sávot ugyanazon a mintán választottuk.
    EN: Repeated stratified k-fold: in every split the band is chosen on the TRAINING folds
        (max macro-F1 on the grid) and applied to the HELD-OUT fold. In each repeat every one of
        the 399 reviews gets exactly one held-out label; the metrics are computed from these and
        averaged over the repeats. The measured performance is thus not affected by the band
        having been chosen on the same sample.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    grid = grid or BAND_GRID
    data = _gold_with_labels()
    pol, y = data["polarity"].to_numpy(), data["gold_label"].to_numpy()

    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    preds: dict[int, np.ndarray] = {}
    chosen = []
    for i, (train, test) in enumerate(splitter.split(pol, y)):
        band = grid[int(np.argmax([macro_f1(y[train], _band_labels(pol[train], b)) for b in grid]))]
        chosen.append(band)
        preds.setdefault(i // n_splits, np.empty(len(y), dtype=object))[test] = _band_labels(pol[test], band)

    per_repeat = pd.DataFrame([{
        "f1_macro": macro_f1(y, p), "accuracy": accuracy_score(y, p), "cohen_kappa": cohen_kappa_score(y, p),
        "recall_negative": float(((p == "negative") & (y == "negative")).sum() / (y == "negative").sum()),
    } for p in preds.values()])
    per_repeat.index.name = "repeat"

    fixed = _band_labels(pol, CFG.review_neutral_band)
    summary = pd.DataFrame({
        "in-sample (band fixed at %.2f)" % CFG.review_neutral_band: {
            "f1_macro": macro_f1(y, fixed), "accuracy": accuracy_score(y, fixed),
            "cohen_kappa": cohen_kappa_score(y, fixed),
            "recall_negative": float(((fixed == "negative") & (y == "negative")).sum() / (y == "negative").sum())},
        f"nested {n_splits}-fold x{n_repeats}: mean": per_repeat.mean().to_dict(),
        f"nested {n_splits}-fold x{n_repeats}: min": per_repeat.min().to_dict(),
        f"nested {n_splits}-fold x{n_repeats}: max": per_repeat.max().to_dict(),
    }).T.round(4)
    chosen_share = pd.Series(chosen).value_counts(normalize=True).sort_index().rename("share_of_folds").round(3)
    chosen_share.index.name = "band chosen on the training folds"
    log.info("Nested band CV: macro-F1 %.3f (in-sample %.3f)", per_repeat["f1_macro"].mean(),
             macro_f1(y, fixed))
    return {"06_band_nested_cv": summary, "06_band_nested_cv_repeats": per_repeat.round(4),
            "06_band_nested_cv_chosen": chosen_share.to_frame()}


# ==========================================================================
# 3) A 9–10 PONTOS VÉLEMÉNYEK PANASZAI / COMPLAINTS IN 9–10 REVIEWS
# ==========================================================================
def high_score_complaints(units: pd.DataFrame | None = None, meta: pd.DataFrame | None = None,
                          n_examples: int = 25, seed: int = CFG.random_seed) -> dict[str, pd.DataFrame]:
    """
    HU: A 01-es notebook "48,1%"-os mutatójának ellenőrzése. A címkézett véleményekből a 9–10
        pontosakat nézzük, amelyekben a panaszmező legalább háromszavas, és három lépcsőt
        számolunk: (a) a teljes panaszmező "nothing"-típusú (non_complaint_mask), (b) van legalább
        egy valódi panasz-tagmondat, (c) van legalább egy valódi panasz-tagmondat, amelyet a (B)
        is negatívnak olvas — ez a panaszelemzés által ténylegesen számolt eset.
    EN: Check of the "48.1%" figure of notebook 01. Among the labelled reviews we take those
        scored 9–10 whose complaint field has at least three words, and compute three steps:
        (a) the whole complaint field is a "nothing"-type formula (non_complaint_mask), (b) there
        is at least one real complaint clause, (c) there is at least one real complaint clause
        that (B) also reads as negative — the case the complaint analysis actually counts.
    """
    from .bert_labelling import load_labels
    from .preprocessing import non_complaint_mask

    meta = _review_meta() if meta is None else meta
    labelled = meta.loc[meta.index.intersection(load_labels("labels_B").index)].copy()
    high = labelled[labelled["score"].round() >= HIGH_SCORE_MIN].copy()
    words = high["review_negative"].fillna("").str.split().str.len()
    with_text = high[words >= MIN_COMPLAINT_WORDS].copy()
    with_text["field_is_formula"] = non_complaint_mask(with_text["review_negative"]).to_numpy()

    clauses = _complaint_units(units)
    clauses = clauses[clauses["review_id"].isin(with_text.index)]
    per_review = clauses.groupby("review_id").agg(real=("non_complaint", lambda s: int((~s).sum())),
                                                  counted=("counted", "sum"))
    with_text = with_text.join(per_review)
    with_text[["real", "counted"]] = with_text[["real", "counted"]].fillna(0).astype(int)
    with_text["has_real_clause"] = with_text["real"] > 0
    with_text["confirmed"] = with_text["counted"] > 0

    n_high, n_text = len(high), len(with_text)
    summary = pd.DataFrame([
        ("reviews scored 9–10 (labelled set)", n_high, 1.0),
        (f"... with a complaint field of ≥{MIN_COMPLAINT_WORDS} words", n_text, n_text / n_high),
        ("... of which the whole field is a 'nothing'-type formula", int(with_text["field_is_formula"].sum()),
         with_text["field_is_formula"].mean()),
        ("... with ≥1 real complaint clause (rule only)", int(with_text["has_real_clause"].sum()),
         with_text["has_real_clause"].sum() / n_high),
        ("... with ≥1 real clause also read as negative by (B) = confirmed", int(with_text["confirmed"].sum()),
         with_text["confirmed"].sum() / n_high),
    ], columns=["step", "n", "share"]).set_index("step")
    summary["share_of_reviews_with_text"] = summary["n"] / n_text
    summary.loc["reviews scored 9–10 (labelled set)", "share_of_reviews_with_text"] = np.nan
    summary = summary.round(4)

    high["confirmed"] = high.index.isin(with_text.index[with_text["confirmed"]])
    high["has_text"] = high.index.isin(with_text.index)
    by_segment = []
    for seg in SEGMENTS:
        if seg not in high.columns:
            continue
        t = (high.groupby(seg).agg(n=("confirmed", "size"), share_with_text=("has_text", "mean"),
                                   share_confirmed=("confirmed", "mean"))
             .query("n >= 1000").sort_values("share_confirmed", ascending=False))
        by_segment.append(t.assign(segment=seg).reset_index().rename(columns={seg: "value"}))
    by_segment = pd.concat(by_segment, ignore_index=True).set_index(["segment", "value"]).round(4)

    confirmed = with_text[with_text["confirmed"]]
    short = confirmed[confirmed["review_negative"].str.split().str.len() <= 25]
    examples = (short.sample(min(n_examples, len(short)), random_state=seed)
                [["score", "accommodation_type", "review_positive", "review_negative"]])
    log.info("9–10 reviews: %.1f%% with a ≥3-word complaint, %.1f%% confirmed", 100 * n_text / n_high,
             100 * with_text["confirmed"].sum() / n_high)
    return {"06_high_score_complaints": summary, "06_high_score_complaints_by_segment": by_segment,
            "06_high_score_complaint_examples": examples}


# ==========================================================================
# 4) A PANASZMEZŐ LEFEDETTSÉGE / COVERAGE OF THE COMPLAINT FIELD
# ==========================================================================
def complaint_coverage(units: pd.DataFrame | None = None, seed: int = CFG.random_seed) -> dict[str, pd.DataFrame]:
    """
    HU: A panaszmező összes tagmondata: hány "nothing"-típusú, és a többiből mennyit olvas a (B)
        negatívnak (ezeket számolja a panaszelemzés), semlegesnek, illetve pozitívnak. A nem
        számolt valódi tagmondatokból példákat is mentünk — ezek mutatják, mi marad ki.
    EN: All complaint-field clauses: how many are "nothing"-type formulas, and of the rest how
        many (B) reads as negative (these are counted by the complaint analysis), neutral or
        positive. Examples of the real clauses NOT counted are saved too — they show what is missed.
    """
    clauses = _complaint_units(units)
    real = clauses[~clauses["non_complaint"]]
    table = pd.DataFrame({
        "n": [len(clauses), int(clauses["non_complaint"].sum()), len(real),
              *[int((real["B_label"] == c).sum()) for c in LABELS]],
    }, index=["complaint-field clauses", "'nothing'-type formulas (removed)", "real complaint clauses",
              *[f"real clauses read as {c}" for c in LABELS]])
    table["share_of_real"] = (table["n"] / len(real)).where(table.index.str.startswith("real clauses"))
    table.index.name = "complaint field"
    # HU: keverés + head(), nem apply (pandas 3.0) / EN: shuffle + head(), not apply (pandas 3.0)
    missed = (real[real["B_label"] != "negative"].sample(frac=1.0, random_state=seed)
              .groupby("B_label").head(15)[["B_label", "sentence"]].sort_values("B_label"))
    return {"06_complaint_coverage": table.round(4), "06_complaint_coverage_missed_examples": missed}


# ==========================================================================
# 5) ALTÉMÁK A TELJES ADATON / SUBTOPICS ON THE FULL DATA
# ==========================================================================
def _read_topic_check(sentiment: str) -> pd.DataFrame | None:
    path = PATHS.annotation / f"topic_check_{sentiment}_{CFG.topic_version}.xlsx"
    if not path.exists():
        return None
    check = pd.read_excel(path)
    check["fits"] = pd.to_numeric(check["fits"], errors="coerce")
    return check.dropna(subset=["fits"])


def topic_check_by_aspect() -> pd.DataFrame:
    """
    HU: A kézi témaellenőrzés aspektusonként: ellenőrzött mondatok, pontosság, a leggyengébb és
        a legerősebb altéma (a végső név szerint összevonva).
    EN: The manual topic check per aspect: sentences checked, precision, the weakest and the
        strongest subtopic (merged by final name).
    """
    rows = []
    for sentiment in ("negative", "positive"):
        check = _read_topic_check(sentiment)
        if check is None:
            continue
        names = pd.read_csv(PATHS.tables / f"05_{sentiment}_{CFG.topic_version}_topic_precision.csv",
                            index_col=0)[["aspect", "topic", "label"]]
        check = check.merge(names, on=["aspect", "topic"], how="left")
        for aspect, d in check.groupby("aspect"):
            per_topic = d.groupby("label")["fits"].mean().sort_values()
            rows.append({"sentiment": sentiment, "aspect": aspect, "checked": len(d),
                         "precision": d["fits"].mean(), "n_subtopics": per_topic.size,
                         "weakest_subtopic": per_topic.index[0], "weakest_precision": per_topic.iloc[0],
                         "subtopics_below_0.70": int((per_topic < 0.70).sum())})
        rows.append({"sentiment": sentiment, "aspect": "(all four)", "checked": len(check),
                     "precision": check["fits"].mean()})
    return pd.DataFrame(rows).set_index(["sentiment", "aspect"]).round(3)


def subtopic_counts(sentiment: str) -> pd.DataFrame:
    """
    HU: Altémánként a TELJES adaton hozzárendelt tagmondatok száma és aránya az aspektuson belül
        (a nevezőben a sehová nem sorolt "(unassigned)" egységek is benne vannak, így az arányok
        összege 100%). Mellé: a mintán (5 000 mondat) mért arány és — ha volt — a kézi pontosság.
    EN: Per subtopic, the number of clauses assigned on the FULL data and their share within the
        aspect (the denominator includes the "(unassigned)" units, so the shares add up to 100%).
        Next to it: the share measured on the fit sample (5,000 sentences) and — if checked —
        the manual precision.
    """
    from .data_loading import read_frame

    assignments = read_frame(PATHS.labels / f"subtopic_assignments_{sentiment}_{CFG.topic_version}")
    total = assignments.groupby("aspect").size().rename("aspect_total")
    counts = (assignments.groupby(["aspect", "topic_label"]).size().rename("n").reset_index()
              .join(total, on="aspect"))
    counts["share"] = counts["n"] / counts["aspect_total"]
    named = pd.read_csv(PATHS.tables / f"05_{sentiment}_{CFG.topic_version}_named_subtopics.csv", index_col=0)
    counts = counts.merge(named[["aspect", "label", "share_within_aspect"]]
                          .rename(columns={"label": "topic_label", "share_within_aspect": "share_fit_sample"}),
                          on=["aspect", "topic_label"], how="left")
    precision_path = PATHS.tables / f"05_{sentiment}_{CFG.topic_version}_topic_precision.csv"
    if precision_path.exists():
        prec = pd.read_csv(precision_path, index_col=0)
        prec = (prec.assign(hits=prec["precision"] * prec["checked"])
                .groupby(["aspect", "label"]).agg(hits=("hits", "sum"), checked=("checked", "sum")))
        prec["manual_precision"] = prec["hits"] / prec["checked"]
        counts = counts.merge(prec[["manual_precision"]].reset_index()
                              .rename(columns={"label": "topic_label"}), on=["aspect", "topic_label"], how="left")
    counts = counts.sort_values(["aspect_total", "n"], ascending=[False, False])
    return counts.set_index(["aspect", "topic_label"]).round(4)


def subtopic_examples(sentiment: str, aspects=KEY_ASPECTS, n_per_subtopic: int = 5,
                      min_words: int = 4, max_words: int = 14, seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Rövid példamondatok altémánként. Ha az aspektust kézzel ellenőriztük, CSAK az illeszkedőnek
        jelölt (fits = 1) mondatokból választunk ("verified"); különben a modell illesztési mintájából
        ("model").
    EN: Short example sentences per subtopic. If the aspect was checked by hand, ONLY sentences
        marked as fitting (fits = 1) are used ("verified"); otherwise the model's fit sample ("model").
    """
    from .data_loading import read_frame

    assignments = read_frame(PATHS.labels / f"subtopic_assignments_{sentiment}_{CFG.topic_version}")
    check = _read_topic_check(sentiment)
    rows = []
    for aspect in aspects:
        sub = assignments[(assignments["aspect"] == aspect) & (assignments["topic"] >= 0)]
        for label in sub["topic_label"].value_counts().index:
            topics = sub.loc[sub["topic_label"] == label, "topic"].unique()
            if check is not None and aspect in set(check["aspect"]):
                pool = check[(check["aspect"] == aspect) & check["topic"].isin(topics) & (check["fits"] == 1)]
                source = "verified"
            else:
                pool = sub[(sub["topic_label"] == label) & sub["in_fit_sample"]]
                source = "model"
            pool = pool[pool["sentence"].str.split().str.len().between(min_words, max_words)]
            pool = pool.sample(min(n_per_subtopic, len(pool)), random_state=seed)
            rows += [{"aspect": aspect, "subtopic": label, "source": source, "sentence": s}
                     for s in pool["sentence"]]
    return pd.DataFrame(rows).set_index(["aspect", "subtopic"])


# ==========================================================================
# 6) PANASZTÉRKÉPEK A KULCSMEGÁLLAPÍTÁSOKHOZ / COMPLAINT MAPS FOR THE KEY FINDINGS
# ==========================================================================
def key_maps(sentiment: str = "negative", aspects=("bathroom", "noise", "value_for_money", "staff",
                                                     "reception", "room"),
             meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    HU: A topic_modeling.segment_table ugyanazokkal a beállításokkal (min. 50 egység, 8 szegmens,
        z-próba + BH, |lift − 1| ≥ 0,2), mint a 05-ös notebook — itt a mentett hozzárendelésekből,
        a szegmenst review_id alapján csatolva. Csak a "notable" cellák kerülnek a táblába.
    EN: topic_modeling.segment_table with the same settings as notebook 05 (min. 50 units,
        8 segments, z-test + BH, |lift − 1| ≥ 0.2) — here from the saved assignments, with the
        segment attached via review_id. Only the "notable" cells are kept.
    """
    from . import topic_modeling as tm
    from .data_loading import read_frame

    meta = _review_meta() if meta is None else meta
    assignments = read_frame(PATHS.labels / f"subtopic_assignments_{sentiment}_{CFG.topic_version}")
    # HU: segment_table a "row" pozícióval indexel -> soronként rekonstruáljuk a szegmenst.
    # EN: segment_table indexes by the "row" position -> rebuild the segment per row.
    row_ids = assignments.drop_duplicates("row").set_index("row")["review_id"]
    ids = np.full(int(assignments["row"].max()) + 1, None, dtype=object)
    ids[row_ids.index.to_numpy()] = row_ids.to_numpy()
    sentences = pd.DataFrame({seg: meta[seg].reindex(ids).to_numpy() for seg in SEGMENTS if seg in meta.columns})

    frames = []
    for aspect in aspects:
        for seg in sentences.columns:
            lift = tm.segment_table(assignments, sentences, aspect, seg, min_count=50, top_segments=8)
            if lift.empty:
                continue
            long = tm.segment_long_table(lift)
            frames.append(long[long["notable"]].assign(aspect=aspect, segment_type=seg))
    out = pd.concat(frames, ignore_index=True)
    return (out[["aspect", "segment_type", "segment", "subtopic", "lift", "count", "segment_n", "q"]]
            .sort_values(["aspect", "segment_type", "lift"], ascending=[True, True, False])
            .set_index(["aspect", "segment_type", "segment", "subtopic"]))


# ==========================================================================
# 7) OPCIONÁLIS KÉZI ELLENŐRZÉS / OPTIONAL MANUAL CHECKS
# ==========================================================================
HIGH_CHECK_FILE = "high_score_complaint_check.xlsx"
EXTRA_TOPIC_CHECK_FILE = f"topic_check_negative_{CFG.topic_version}_bathroom_noise.xlsx"


def export_manual_checks(n_high: int = 100, n_per_topic: int = 20, units: pd.DataFrame | None = None,
                         meta: pd.DataFrame | None = None, seed: int = CFG.random_seed) -> list:
    """
    HU: Két OPCIONÁLIS kézi ellenőrző fájl (ha még nem léteznek):
          • 100 véletlen 9–10 pontos vélemény panaszmezője (≥3 szó) — "genuine" oszlop: 1 = valódi
            elégedetlenség, 0 = nem az (pl. "nothing to complain about"). VAK: sem a pontszám,
            sem a modell döntése nem látszik.
          • a fürdőszoba és a zaj negatív altémái, témánként 20 mondat — "fits" oszlop, mint a
            05-ös notebook ellenőrzésénél.
    EN: Two OPTIONAL manual check files (if they do not exist yet):
          • the complaint field (≥3 words) of 100 random 9–10 reviews — "genuine" column: 1 = real
            dissatisfaction, 0 = not (e.g. "nothing to complain about"). BLIND: neither the score nor
            the model's decision is shown.
          • the negative subtopics of bathroom and noise, 20 sentences per topic — "fits" column, as
            in the check of notebook 05.
    """
    from .bert_labelling import load_labels
    from .data_loading import read_frame

    PATHS.ensure()
    written = []
    path = PATHS.annotation / HIGH_CHECK_FILE
    if not path.exists():
        meta = _review_meta() if meta is None else meta
        labelled = meta.loc[meta.index.intersection(load_labels("labels_B").index)]
        high = labelled[(labelled["score"].round() >= HIGH_SCORE_MIN)
                        & (labelled["review_negative"].fillna("").str.split().str.len() >= MIN_COMPLAINT_WORDS)]
        sample = high.sample(min(n_high, len(high)), random_state=seed)
        pd.DataFrame({"review_id": sample.index, "complaint_field": sample["review_negative"].to_numpy(),
                      "genuine": ""}).to_excel(path, index=False)
        written.append(path)
        log.info("Exported: %s (%d complaint fields) — fill in 'genuine' (1/0)", path.name, len(sample))

    path = PATHS.annotation / EXTRA_TOPIC_CHECK_FILE
    if not path.exists():
        assignments = read_frame(PATHS.labels / f"subtopic_assignments_negative_{CFG.topic_version}")
        sample = (assignments[assignments["aspect"].isin(["bathroom", "noise"]) & (assignments["topic"] >= 0)]
                  .sample(frac=1.0, random_state=seed)
                  .groupby(["aspect", "topic"], sort=False).head(n_per_topic)
                  .sort_values(["aspect", "topic"]))
        sample.assign(fits="")[["aspect", "topic", "topic_label", "sentence", "fits"]].to_excel(path, index=False)
        written.append(path)
        log.info("Exported: %s (%d sentences) — fill in 'fits' (1/0)", path.name, len(sample))
    return written


def evaluate_manual_checks() -> dict[str, pd.DataFrame]:
    """HU: A kitöltött opcionális ellenőrzések kiértékelése. / EN: Evaluate the filled-in optional checks."""
    out = {}
    path = PATHS.annotation / HIGH_CHECK_FILE
    if path.exists():
        check = pd.read_excel(path)
        check["genuine"] = pd.to_numeric(check["genuine"], errors="coerce")
        check = check.dropna(subset=["genuine"])
        if len(check):
            k, n = int(check["genuine"].sum()), len(check)
            low, high = _wilson(k, n)
            out["06_high_score_manual_check"] = pd.DataFrame(
                {"checked": [n], "genuine": [k], "share": [k / n], "ci95_low": [low], "ci95_high": [high]}).round(4)
    path = PATHS.annotation / EXTRA_TOPIC_CHECK_FILE
    if path.exists():
        check = pd.read_excel(path)
        check["fits"] = pd.to_numeric(check["fits"], errors="coerce")
        check = check.dropna(subset=["fits"])
        if len(check):
            per_topic = (check.groupby(["aspect", "topic_label"])["fits"].agg(checked="size", precision="mean"))
            per_aspect = check.groupby("aspect")["fits"].agg(checked="size", precision="mean")
            out["06_bathroom_noise_topic_precision"] = per_topic.round(3)
            out["06_bathroom_noise_precision_by_aspect"] = per_aspect.round(3)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """HU: Wilson-féle 95%-os intervallum egy arányra. / EN: Wilson 95% interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# ==========================================================================
# FUTTATÁS / RUN
# ==========================================================================
CHECKS = ("gold", "band", "high", "coverage", "subtopics", "maps", "export")


def run(checks=CHECKS) -> dict[str, pd.DataFrame]:
    """HU: A kiválasztott ellenőrzések futtatása és mentése. / EN: Run and save the selected checks."""
    from .data_loading import read_frame
    from .evaluation import save_table

    checks = set(checks)
    units = read_frame(PATHS.labels / "field_units_B") if checks & {"gold", "high", "coverage"} else None
    meta = _review_meta() if checks & {"high", "maps", "export"} else None
    tables: dict[str, pd.DataFrame] = {}
    if "gold" in checks:
        tables.update(gold_tables(units))
    if "band" in checks:
        tables.update(band_nested_cv())
    if "high" in checks:
        tables.update(high_score_complaints(units, meta))
    if "coverage" in checks:
        tables.update(complaint_coverage(units))
    if "subtopics" in checks:
        tables["06_topic_check_by_aspect"] = topic_check_by_aspect()
        for sentiment in ("negative", "positive"):
            tables[f"06_subtopic_counts_{sentiment}"] = subtopic_counts(sentiment)
        tables["06_subtopic_examples_negative"] = subtopic_examples("negative")
        tables["06_subtopic_examples_positive"] = subtopic_examples("positive", aspects=POSITIVE_KEY_ASPECTS)
    if "maps" in checks:
        tables["06_key_maps_negative"] = key_maps("negative", meta=meta)
    if "export" in checks:
        export_manual_checks(units=units, meta=meta)
        tables.update(evaluate_manual_checks())
    for name, table in tables.items():
        save_table(table, name)
    return tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Checks requested in the supervisor's feedback")
    parser.add_argument("checks", nargs="*", choices=CHECKS, default=None,
                        help="which checks to run (default: all)")
    args = parser.parse_args(argv)
    PATHS.ensure()
    run(args.checks or CHECKS)


if __name__ == "__main__":
    main()
