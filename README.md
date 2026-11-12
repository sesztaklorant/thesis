# Text-based sentiment labelling of hotel reviews with BERT

Software project of the BSc thesis *Sentiment Analysis of Customer Reviews Using Machine Learning Techniques: Sentiment Analysis of Hotel Guest Reviews with BERT-Based Labelling and Transformer Models* (Corvinus University of Budapest, 2026).

The program labels Booking.com reviews from their text instead of the guest score, validates the labels and adds a thematic layer that shows what guests complain about and what they praise.

| Layer | Content | Notebook |
|---|---|---|
| 1 | sentiment label and 1–10 score from the text with three BERT approaches: (A) zero-shot, (B) transfer learning on HRAST, (C) self-training | 02, 03 |
| 2/a | aspect of each sentence (room, staff, bathroom, noise, ...) with a multi-label BERT trained on HRAST | 05 |
| 2/b | subtopics within each aspect (Sentence-BERT, UMAP, HDBSCAN, c-TF-IDF) and complaint maps by segment | 05 |

## Folder structure

```
CODE/
├── data/
│   ├── raw/
│   │   ├── booking/     val.csv of the Booking.com dataset + README.md
│   │   └── hrast/       HRAST csv + README.md
│   ├── interim/         cache of the loaded csv
│   ├── processed/       booking_prepared, booking_working_set, booking_labelled
│   ├── labels/          labels of (A), (B), (C), final label, 2/a and 2/b outputs; cache/ for resumable runs
│   └── annotation/      gold sample and manual topic checks
├── notebooks/           01–05 in run order
├── src/                 source code (16 modules)
├── tests/               offline tests
├── results/
│   ├── tables/          CSV outputs, prefixed with the notebook number (01_ ... 06_)
│   ├── figures/         PNG outputs
│   └── models/          trained models (not in Git)
├── docs/                ANNOTATION_GUIDE.md, DEVELOPMENT_LOG.md
├── archive/             earlier notebooks, outputs and working documents, not used by the final run
└── requirements.txt
```

## Data

The datasets are not part of the repository because of their size and licences.

| Dataset | Use | Folder | Source | Licence |
|---|---|---|---|---|
| Booking.com Accommodation Reviews (Igebaria et al., 2024) | analysed data, `val.csv` only (203,787 reviews) | `data/raw/booking/` | [Hugging Face](https://huggingface.co/datasets/Booking-com/accommodation-reviews) | CC BY-SA 4.0 |
| HRAST (Andreou et al., 2023) | training data of labeller (B) and of the 2/a aspect model | `data/raw/hrast/` | [Kaggle](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) | CC BY-NC 4.0 |

The README.md in each folder is the description published with the dataset. If the HRAST csv is missing, `data_loading.load_hrast_raw()` tries to download it with `kagglehub`.

## Environment

Developed on Windows 11 with Python 3.12.3. Fine-tuning ran on an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM). Notebooks 01, 03 and 04 also run on a CPU.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

For GPU use, PyTorch is installed as a CUDA build from pytorch.org. The build used in the project:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On a 4 GB GPU the training uses a batch of 4 with 8 accumulation steps (effective batch 32), on a Colab T4 a batch of 32 (`local_batch_size`, `local_grad_accum` and `batch_size` in `src/config.py`).

## Run order

| # | Notebook | Hardware | Content |
|---|---|---|---|
| 1 | `01_data_preparation_eda` | CPU | loading, data quality, cleaning, exploratory analysis, working set |
| 2 | `02_bert_labelling` | GPU | labelling approaches (A), (B), (C), field test, final label |
| 3 | `03_label_validation` | CPU | agreement, guest score, field test, HRAST test, gold sample |
| 4 | `04_text_analysis_classical` | CPU | distinctive terms, cross-tabulations, keyword aspects, TF-IDF student models |
| 5 | `05_thematic_layer` | GPU recommended | 2/a aspect model, 2/b subtopics, complaint maps, manual check |
| 6 | `python -m src.feedback_checks` | CPU | additional checks of the existing outputs (`results/tables/06_*.csv`) |

Notebook 05 runs once with `SENTIMENT = "negative"` and once with `SENTIMENT = "positive"`. Steps 1–2 can also be run from the command line (`python -m src.pipeline all`, the individual steps are listed in `src/pipeline.py`).

Models (B), (C) and the 2/a aspect model are trained once and reused when they exist in `results/models/` (`RETRAIN_B = False`, `RETRAIN_C = False` in notebook 02). Predictions are saved in chunks under `data/labels/cache/`, so an interrupted notebook continues from the last saved chunk when it is run again.

Notebook 02 can also run on Google Colab: with the `CODE` folder in `MyDrive/THESIS/CODE`, the first cell mounts Google Drive and sets `PROJECT_DIR`.

## Final settings

All settings are in `src/config.py`. The final run uses the full data (`labelling_sample_size = None`), the neutral band 0.25, labeller (B) as the final label and the subtopic version `topic_version = "full2"`. The 2/b output names contain the version, and the manual subtopic names in `src/topic_names.py` belong to `full2`.

## Tests

```
python -m pytest tests/
```

19 tests (`test_offline.py`, `test_feedback_checks.py`) check the labelling, aggregation, validation and feedback check logic with a keyword-based fake model, without GPU, internet or PyTorch.

## Versions

| Git tag | State |
|---|---|
| `v0.3-sample20k` | development run on a random sample of 20,000 reviews |
| `v1.0-full` | full run on 203,328 reviews, manual check and subtopic names |
| `v1.1-final` | final version: `v1.0-full` with the additional checks (`src/feedback_checks.py`) and the cleaned-up code |

The development steps are summarised in `docs/DEVELOPMENT_LOG.md`.

## Conventions

- Code, comments and column names are in English.
- Settings are defined only in `src/config.py`.
- The notebooks contain the analysis steps and call the functions of `src/`.
