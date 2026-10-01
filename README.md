# Hotelvélemények szövegalapú szentiment-címkézése BERT-tel
# Text-based sentiment labelling of hotel reviews with BERT

Pályamű típusú szakdolgozat kódja / Code of a software-project type thesis.
Részletes terv / Detailed plan: [`docs/PALYAMU_TERV.md`](docs/PALYAMU_TERV.md)

## Mappaszerkezet / Folder structure

```
CODE/
├── data/
│   ├── raw/            val.csv (Booking) + hrast/ (HRAST CSV)          ← HU: ide másold / EN: put them here
│   ├── processed/      booking_prepared, booking_working_set, booking_labelled
│   ├── labels/         labels_A / _B / _C / _final + cache/ (folytatható futás / resumable runs)
│   └── annotation/     gold_sample_to_annotate.csv (kézi címke / manual labels)
├── results/            figures/, tables/, models/
├── src/                a program forráskódja / source code of the program
├── notebooks/          01 → 02 (GPU) → 03 → 04
├── tests/              python tests/test_offline.py
└── docs/               PALYAMU_TERV.md, ANNOTATION_GUIDE.md
```

## Futtatási sorrend / Run order

| # | Notebook | Hol? / Where? | Mit csinál? / What does it do? |
|---|---|---|---|
| 1 | `01_data_preparation_eda` | helyben / locally | betöltés, tisztítás, EDA, munkahalmaz / load, clean, EDA, working set |
| 2 | `02_bert_labelling` | **Colab GPU** | (A) zero-shot, (B) HRAST, (C) önképzés / self-training → végső címke / final label |
| 3 | `03_label_validation` | helyben / locally | egyezés, pontszám, mezőpróba, gold minta / agreement, score, field test, gold sample |
| 4 | `04_text_analysis_classical` | helyben / locally | TF-IDF szóelemzés, kereszttáblák, aspektusok, klasszikus modellek |

Parancssorból ugyanez / The same from the command line: `python -m src.pipeline all`

## Colab

1. Töltsd fel a teljes `CODE` mappát a Google Drive-ra (`MyDrive/THESIS/CODE`). / Upload the whole `CODE` folder to Drive.
2. Nyisd meg a `notebooks/02_bert_labelling.ipynb`-t Colabban, *Runtime → T4 GPU*. / Open notebook 02 in Colab.
3. Az első cellában állítsd be a `PROJECT_DIR`-t. / Set `PROJECT_DIR` in the first cell.
4. Minden eredmény a Drive-ra íródik; megszakadt futás újraindítva folytatódik. / Everything is written to Drive; an interrupted run resumes.

## Kódolási konvenciók / Coding conventions

- Kommentek és docstringek: **magyarul és angolul** (`HU:` / `EN:`). / Comments and docstrings: **Hungarian and English**.
- Változó-, függvény- és oszlopnevek: **csak angolul**. / Variable, function and column names: **English only**.
- Minden beállítás a `src/config.py`-ban. / Every setting lives in `src/config.py`.
