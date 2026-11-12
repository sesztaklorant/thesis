# Annotation guide for the gold sample

File: `data/annotation/gold_sample_to_annotate.xlsx` (400 randomly selected reviews, exported by notebook 03).

## Rules

1. Only the review text is read. The file deliberately contains no guest score and no model label, so the manual label stays independent of both.
2. The label answers one question: what overall impression of the stay does the text give?

| `gold_label` | When | Example |
|---|---|---|
| `positive` | mostly praise, any complaint is minor | "Great location, lovely staff. Pillows a bit soft." |
| `negative` | mostly complaints | "Dirty room, rude staff. Location was ok." |
| `neutral` | factual or lukewarm, or praise and complaint roughly balanced | "Room fine. Breakfast average." / "Amazing view but the room was filthy." |

3. `gold_score` is an integer from 1 to 10 that reflects how satisfied the writer is (1 = very dissatisfied, 10 = fully satisfied). A neutral text usually gets 5 or 6.
4. Texts that cannot be interpreted (for example not in English) are left empty and are skipped by `load_gold()`.
5. The annotation is done in blocks of at most about 100 reviews, with breaks in between, to limit the effect of fatigue.

## Possible extension

If a second person independently annotated 100 of the same reviews, Cohen's kappa between the two annotators would give the human upper limit of the task.
