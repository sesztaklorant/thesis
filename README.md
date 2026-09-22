# Sentiment Analysis of Customer Reviews using Machine Learning Techniques

Szakdolgozat-projekt — hotel-vendégértékelések sentiment- és aspektus-alapú
elemzése klasszikus gépi tanulással és transformer modellekkel.

**Témavezető:** Mutaz AlShafeey

---

## Hol kezdd?

| Ha ezt akarod | Olvasd el ezt |
|---|---|
| Beállítani a gépedet | **[KORNYEZET.md](KORNYEZET.md)** ← kezdd itt |
| Megérteni, mi lesz a terv | **[MUNKATERV.md](MUNKATERV.md)** |
| Elkezdeni a munkát | `notebooks/01_data_loading_eda.ipynb` |

**Gyorsindítás:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tests/test_pipeline.py      # ennek zöldnek kell lennie
```

---

## Projektszerkezet

```
hotel-absa/
├── KORNYEZET.md            ← környezet beállítása lépésről lépésre
├── MUNKATERV.md            ← fázisok, időterv, dolgozat-fejezet megfeleltetés
├── requirements.txt
│
├── src/                    ← újrafelhasználható kód (a notebookok ezt hívják)
│   ├── config.py           ← MINDEN beállítás egy helyen (seed, küszöbök, utak)
│   ├── data_loading.py     ← a 4 adathalmaz betöltése egységes formára
│   ├── preprocessing.py    ← tisztítás, címkézés, 60/20/20 felosztás
│   ├── models_classical.py ← TF-IDF + NB / LogReg / SVM + abláció
│   ├── evaluation.py       ← metrikák, confusion matrix, hibaelemzés
│   └── aspects.py          ← ABSA: mondatra bontás, aspektus-lexikon
│
├── notebooks/
│   ├── 01_data_loading_eda.ipynb    ← adat + EDA          (4., 6.1, 9. fejezet)
│   ├── 02_classical_baseline.ipynb  ← TF-IDF modellek     (RQ1 fele, RQ2)
│   └── 03_bert_colab.ipynb          ← BERT (Colab, GPU)   (RQ1 másik fele)
│
├── data/                   ← Git-ből kizárva (méret + licenc)
│   ├── raw/                ← érintetlen letöltés
│   └── processed/          ← modellezésre kész, splitelt parquet
│
├── results/                ← IDE KERÜL MINDEN, AMI A DOLGOZATBA MEGY
│   ├── figures/            ← 300 DPI ábrák
│   ├── tables/             ← CSV táblázatok
│   └── models/             ← mentett modellek (Git-ből kizárva)
│
├── docs/naplo.md           ← kutatási napló — ebből lesz a dolgozat szövege
└── tests/test_pipeline.py  ← füstteszt szintetikus adaton
```

---

## Kutatási kérdések és hol válaszoljuk meg őket

| RQ | Kérdés | Hol |
|---|---|---|
| **RQ1** | Klasszikus ML vs. transformer 3-osztályos sentimentben | `02` + `03` notebook |
| **RQ2** | Mennyit számít az előfeldolgozás és a TF-IDF reprezentáció | `02` — ablációs vizsgálat |
| **RQ3** | Mely hotel-aspektusok kötődnek mely sentimenthez | `01` (baseline) + `04` (transformer) |
| **RQ4** | Transformer-alapú aspektus-szintű elemzés és bizonyíték-kiemelés | `04` notebook |
| **RQ5** | Általánosítás más adathalmazra | `02` és `03` utolsó szakasza |

---

## Adathalmazok

| Név | Szerep | Méret | Licenc |
|---|---|---|---|
| [Booking.com Accommodation Reviews](https://huggingface.co/datasets/Booking-com/accommodation-reviews) | fő | ~1,6M | CC BY-SA 4.0 (nem kereskedelmi) |
| [TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) | külső validáció | ~20k | CC BY-NC-SA 4.0 |
| [515K Hotel Reviews in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) | tartalék / 2. validáció | ~515k | Kaggle — a leírás szerint |
| [HRAST](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) | ABSA | 23 113 mondat | ellenőrizendő |

> Az adatfájlok **nincsenek** a repóban. A `src/data_loading.py` letölti őket,
> vagy kézzel a `data/raw/` mappába teheted.

---

## Reprodukálhatóság

- Minden véletlen `CFG.random_seed = 42`-vel rögzítve (`set_seed()` a notebookok első cellájában)
- Minden beállítás a `src/config.py`-ban, sehol nincs beégetett „varázsszám"
- A `config.py` pillanatképe a splitelt adat mellé mentődik (`config_snapshot.json`)
- A TF-IDF és az osztályozó egy `Pipeline`-ban → nincs adatszivárgás a teszthalmazból
- `requirements.txt` rögzíti a csomagverziókat
