# Development log

Summary of the main development steps and the problems that triggered them. Outputs of earlier versions are kept in `archive/`.

## September 2026: first plan with score-based labels

The first version was organised around research questions on several datasets (Booking.com, TripAdvisor, 515K European hotel reviews), and the sentiment label was a cut of the guest score (at most 7 negative, 8 neutral, at least 9 positive). The classical baseline on this label (TF-IDF and logistic regression, `archive/notebooks/002_classical_baseline.ipynb`) reached a macro-F1 of 0.556.

The guest score rates the whole stay rather than the tone of the text, so the cut is not a valid sentiment label. In the data, 39.7% of the 10-point, 61.4% of the 9-point and 75.6% of the 8-point reviews contain a complaint of at least three words.

## October 2026: labels from the text (20,000-review sample)

The label is produced from the text by three approaches: (A) zero-shot models, (B) BERT fine-tuned on HRAST and (C) self-training on the most confident Booking labels. HRAST, earlier part of an aspect analysis plan only, became the training data of (B). Development ran on a random sample of 20,000 reviews.

| Observed problem | Change |
|---|---|
| Only 0.4% of the argmax labels of (B) were neutral, because averaging the sentence probabilities made mixed reviews (44.9%) positive | neutral band on the polarity, width 0.25 calibrated on the gold sample |
| The neutral class of the three-class model (A) means "no opinion", while a neutral hotel review is usually mixed | (B) as the only teacher of (C) |
| (B) alone was as good on the gold sample as the majority vote of the three approaches | final label = (B) |

The 400-review gold sample was annotated blind (`docs/ANNOTATION_GUIDE.md`), and the evaluation uses the 399 labelled reviews.

## October 2026: thematic layer (2/a and 2/b)

The second layer adds a multi-label aspect classifier trained on HRAST (2/a) and subtopics within each aspect (2/b). The subtopic model went through the following versions; the version tag is part of every 2/b output name (`CFG.topic_version`).

| Version | Observed problem | Change |
|---|---|---|
| early | in sentences of the full text, praise and complaints merged into one unit | clause units, split separately from the praise and complaint fields |
| v1 → v2 | 1% cluster size: 19–21 small, overlapping subtopics per aspect, bathroom stability ARI 0.44 | 3% cluster size, granularity sweep |
| v2 → v3 | 3%: bathroom fell from 18 to 2 subtopics (one cluster with 86%) | fine clustering at 2% and topic merging (at most 10 subtopics, similarity ≥ 0.90) |
| v3 → v4 | centroid-based merging produced 40–55% catch-all topics | average-linkage merging, merged topic at most 35% |
| v4 → v5 | positive subtopics were mixed lists ("staff / friendly / location" 76%) | comma-separated lists in the praise field split into units |
| full → full2 | on the full data, fitting on 15,000 sentences per aspect gave giant clusters (negative staff 97%) | fitting on 5,000 sentences, cluster size halved when one cluster exceeds 50%, the remaining sentences assigned to the nearest subtopic |
| final | pseudo-complaints ("nothing", "didn't like"); random lift spikes in small segments; uninformative automatic labels | non-complaint filter; z-test with Benjamini–Hochberg correction and effect size; manual names |

The development state was tagged `v0.3-sample20k` (Git commit `d9db3e0d`).

## October 2026: full run

The program ran on the full data (203,328 reviews after filtering). Model (B) was not retrained because it learns only from HRAST. Model (C) labelled the full data with the model trained in the development run on 10,744 pseudo-labelled reviews, so the gold sample results did not change.

The manual check covered 2 × 800 clauses (20 per subtopic in the four largest aspects) with a precision of 0.840 for complaints and 0.927 for praise. The 188 negative and 191 positive clusters received 167 + 167 English names (`src/topic_names.py`). This state was tagged `v1.0-full` (commit `f6c9a0a9`).

## November 2026: additional checks

`src/feedback_checks.py` adds checks of the existing outputs without a new model or run: confusion matrix and missed negatives on the gold sample, nested cross-validation of the neutral band, complaints in reviews scored 9–10, coverage of the complaint analysis, subtopic counts and complaint maps for the key findings (`results/tables/06_*.csv`).

## Final cleanup

Data folders `data/raw/booking/` and `data/raw/hrast/` with the dataset descriptions, loader cache moved to `data/interim/`, comments in English only, earlier notebooks, outputs and working documents moved to `archive/`. The final version is tagged `v1.1-final`.
