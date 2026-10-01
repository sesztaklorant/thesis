# Annotálási útmutató / Annotation guide — gold minta / gold sample

Fájl / File: `data/annotation/gold_sample_to_annotate.csv` (Excelben is megnyitható / opens in Excel)

## Szabályok / Rules

1. **Csak a szöveget olvasd.** A fájlban szándékosan nincs vendégpontszám és modellcímke — ne nézd meg őket
   annotálás közben, különben a címkéd nem lesz független.
   **Read only the text.** The file intentionally has no guest score and no model label — do not look them up while
   annotating, otherwise your label is not independent.
2. A kérdés: **milyen összbenyomást ad a szöveg a tartózkodásról?**
   The question: **what overall impression of the stay does the text give?**

| `gold_label` | Mikor? / When? | Példa / Example |
|---|---|---|
| `positive` | túlnyomóan dicsér; az esetleges panasz apróság / mostly praise; any complaint is minor | "Great location, lovely staff. Pillows a bit soft." |
| `negative` | túlnyomóan panaszkodik / mostly complaints | "Dirty room, rude staff. Location was ok." |
| `neutral` | tényközlő, langyos, VAGY a dicséret és a panasz kb. kiegyenlített / factual, lukewarm, OR praise and complaint roughly balanced | "Room fine. Breakfast average." / "Amazing view but the room was filthy." |

3. `gold_score` (1–10, egész / integer): mennyire elégedett a szöveg írója — 1 = nagyon elégedetlen,
   10 = teljesen elégedett. Ha bizonytalan vagy, a semleges szöveg 5–6.
   How satisfied the writer of the text is — 1 = very dissatisfied, 10 = fully satisfied.
4. Ha egy szöveg értelmezhetetlen (pl. nem angol), hagyd üresen — a betöltő kihagyja.
   If a text is uninterpretable (e.g. not English), leave it empty — the loader skips it.
5. Dolgozz egyhuzamban legfeljebb ~100 véleményt, utána tarts szünetet (a fáradtság torzít).
   Work through at most ~100 reviews in one go, then take a break (fatigue biases).

## Ajánlott kiegészítés / Recommended extension

Ha egy második személy (pl. évfolyamtárs) ugyanebből 100 véleményt függetlenül annotál, a kettőtök közötti
Cohen-kappa megadja a **feladat emberi felső korlátját** — ennél jobbat egy modelltől sem lehet várni.
If a second person independently annotates 100 of the same reviews, the Cohen's kappa between you gives the
**human ceiling of the task** — no model can be expected to do better.
