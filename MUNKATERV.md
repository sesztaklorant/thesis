# Munkaterv — a kódtól a kész dolgozatig

**Téma:** Sentiment Analysis of Customer Reviews using Machine Learning Techniques
**Témavezető:** Mutaz AlShafeey
**Állapot:** Milestone 1 elfogadva → kódolási fázis indul

---

## A vezérelv

> **Minden kódfázis egy konkrét kutatási kérdésre válaszol, és egy konkrét
> dolgozat-fejezetbe termel anyagot.**

Ez nem formalitás. A leggyakoribb szakdolgozat-hiba, hogy valaki három hónapig
kódol, aztán ül a kész modellek előtt, és nem tudja, mit írjon. Ha minden
fázis végén **azonnal leírod**, amit találtál, a dolgozat magától összeáll.

Ezért van a projektben egy `docs/naplo.md` — minden fázis után írj bele
néhány bekezdést. Nem kell szépen fogalmazni, csak rögzítsd, mit csináltál,
mit láttál, és miért döntöttél úgy, ahogy.

---

## Adatstratégia — javaslat a Milestone 1-hez képest

A Milestone 1-ben három adathalmaz szerepel, három szereppel. Ezt megtartom,
egy fontos módosítással:

| Szerep | Javasolt adathalmaz | Miért | RQ |
|---|---|---|---|
| **Fő** | Booking.com (Hugging Face) | Hivatalos kiadás, **CC BY-SA 4.0** — tiszta licenchelyzet, amit hivatkozni tudsz. 1,6M review. Két sorból betölthető → a munkád reprodukálható. | RQ1, RQ2 |
| **Külső validáció** | **TripAdvisor** (~20k) | ⚠️ **Itt térek el a Milestone 1-től.** | RQ5 |
| **ABSA** | HRAST (23 113 annotált mondat) | Az egyetlen aspektus-szinten annotált hotel-adat. Nélkülözhetetlen. | RQ3, RQ4 |
| Tartalék | 515K Hotel Reviews (Kaggle) | Ha a Booking HF letöltés akadozik, ugyanaz a kód működik rajta. | — |

### Miért a TripAdvisor a külső validációra?

A Milestone 1 az 515K Kaggle adathalmazt jelöli meg külső validációra. A gond:
**az 515K adat is Booking.com-ról származik**, csak más időszakból és más
scrape-ből. Ha ezen tesztelsz, nem a *platformok közötti* általánosítást méred,
hanem lényegében ugyanazt az eloszlást. Az eredmény jó lesz, de keveset mond.

A TripAdvisor **más platform**: más felhasználói bázis, más értékelési skála
(1–5 csillag a 1–10 helyett), más írásstílus (hosszabb, elbeszélőbb reviewk).
Ha a modelled ezen is megáll, az valódi állítás. Ha romlik, az is érdekes
eredmény — sőt, az általában értékesebb, mert van mit megmagyarázni.

**Javaslat:** futtasd le **mindkettőn**, és tedd be egy táblázatba. Így az
RQ5 válasza árnyalt lesz: „ugyanazon a platformon belül a teljesítmény
megmarad (−2 pont), platformok között viszont jelentősen romlik (−11 pont)".
Ez sokkal erősebb megállapítás, mint bármelyik önmagában.

> **Beszéld meg a témavezetőddel** — ez a Milestone 1-hez képest eltérés,
> és jobb, ha előre egyeztetve van. A kód mindkét változatot támogatja,
> tehát nem kockázat.

---

## Fázisok

### 0. fázis — Környezet (fél nap)

`KORNYEZET.md` végigcsinálása, a füstteszt lefuttatása, Git repó létrehozása.

**Kész, ha:** `python tests/test_pipeline.py` zöld, és van egy első commitod.

---

### 1. fázis — Adat és feltáró elemzés (1 hét)

**Notebook:** `01_data_loading_eda.ipynb`
**Dolgozat-fejezetek:** 4. (Data Sources), 6.1 (Preprocessing), 9. (EDA)

| Feladat | Kimenet |
|---|---|
| Booking.com adat betöltése (először 50k minta) | `data/raw/booking_hf.parquet` |
| Adatminőség-vizsgálat: hiányzó értékek, duplikátumok, hosszeloszlás | naplóbejegyzés |
| Sentiment-címkék képzése pontszámból + **küszöb-érzékenységi vizsgálat** | `01_label_threshold_sensitivity.csv` |
| Szűrés és tisztítás, a szűrési napló rögzítése | táblázat a 6.1 fejezetbe |
| EDA-ábrák | `01_eda_overview.png`, `01_top_words_by_sentiment.png` |
| Kulcsszavas aspektus-előzetes | `01_aspect_sentiment_distribution.csv`, `01_aspect_sentiment_stacked.png` |
| 60/20/20 felosztás és mentés | `data/processed/booking_{train,val,test}.parquet` |

**Amit ebben a fázisban meg kell írnod (naplóba):**

1. Miért ezt az adathalmazt választottad, mi a licenc, és hogyan hivatkozod
2. Hogyan képezted a három osztályt, és **miért éppen 6/8 a küszöb** — az
   érzékenységi táblázattal alátámasztva
3. Mit dobott ki a szűrés, és miért (konkrét számokkal)
4. Mi a feltűnő az eloszlásokban (kiegyensúlyozatlanság, hosszkülönbségek)

**A legfontosabb megfigyelés, amit keresni kell:** mennyire
kiegyensúlyozatlan az adat, és mekkora a neutral osztály. Ez fogja
meghatározni az egész dolgozat értékelési stratégiáját.

---

### 2. fázis — Klasszikus baseline (1–1,5 hét)

**Notebook:** `02_classical_baseline.ipynb`
**Fejezetek:** 6.2 (Classical ML), 8. (Evaluation) | **RQ1 fele, RQ2 egésze**

| Feladat | Kimenet |
|---|---|
| Naive Bayes / LogReg / LinearSVM betanítása TF-IDF-fel | `02_classical_validation_results.csv` |
| A legjobb modell kiértékelése a teszthalmazon | `02_*_test_per_class.csv` |
| Confusion matrixok | `02_confusion_matrices_classical.png` |
| **Ablációs vizsgálat (RQ2)** — 8 előfeldolgozási variáns | `02_preprocessing_ablation.csv/.png` |
| Top jellemzők osztályonként | `02_top_features_logreg.csv/.png` |
| Hibaelemzés, 20–30 példa kézi átolvasása | `02_error_type_frequency.csv` |

**Amire figyelj:**

- A **macro-F1** a fő metrikád, nem az accuracy. Ezt indokold meg a dolgozatban.
- Az ablációban nézd meg külön, hogy a **tagadószavak eldobása rontott-e**.
  Ha igen (valószínű), az egy önálló bekezdést érdemel — jól mutatja, miért
  korlátozott a zsák-a-szavak megközelítés.
- A hibaelemzést **tényleg olvasd el kézzel**. A neutral osztály félreosztályozott
  példái fogják megadni a dolgozat legérdekesebb kvalitatív megfigyeléseit.

**Jegyezd fel a legjobb macro-F1-et** — ez lesz a BERT viszonyítási alapja.

---

### 3. fázis — BERT finomhangolás (1–1,5 hét)

**Notebook:** `03_bert_colab.ipynb` (Google Colab, T4 GPU)
**Fejezetek:** 6.3 (Transformer) | **RQ1 másik fele**

| Feladat | Kimenet |
|---|---|
| Parquet fájlok feltöltése Colabra | — |
| `bert-base-uncased` finomhangolása (3 epoch, ~20 perc) | betanított modell |
| Tanulási görbe | `03_bert_training_curve.png` |
| Teszt-kiértékelés, confusion matrix | `03_bert_confusion_matrix.png` |
| **RQ1 összehasonlító táblázat** (4 modell × metrikák × tanítási idő) | `03_rq1_model_comparison.csv` |
| *(opcionális)* DistilBERT és/vagy RoBERTa | plusz sorok a táblázatban |

**A fázis kulcsa — ne csak azt írd le, melyik modell jobb:**

Írd le, hogy **mennyivel és milyen áron**. A klasszikus modellek másodpercek
alatt tanulnak egy laptopon; a BERT percekig egy GPU-n. Ha a különbség
5 pont macro-F1, akkor egy valódi üzleti döntésnél ez mérlegelés kérdése.
Ez a fajta trade-off elemzés az, ami egy jó szakdolgozatot megkülönböztet
egy „lefuttattam három modellt" dolgozattól.

Érdemes a DistilBERT-et is berakni: általában a BERT pontosságának ~97%-át
hozza a futásidő 40%-áért. Egy sor a táblázatban, és lesz egy erős
hatékonysági érved.

---

### 4. fázis — ABSA (2 hét, ez a legnagyobb falat)

**Notebook:** `04_absa_transformer.ipynb` *(ezt együtt írjuk meg, amikor ideérsz)*
**Fejezetek:** 7. (ABSA), 10. (Contribution) | **RQ3, RQ4**

| Feladat | Megjegyzés |
|---|---|
| HRAST letöltése, **oszlopnevek ellenőrzése** | a `load_hrast()` kiírja őket |
| Az annotációs séma megértése és dokumentálása | ez fél nap, de elkerülhetetlen |
| Az aspektus-lexikon illesztése a HRAST aspektusaihoz | `src/aspects.py` |
| Baseline: kulcsszavas aspektusdetektálás + sentiment | már megvan |
| **Sentence-pair BERT ABSA** betanítása HRAST-on | `[CLS] aspect [SEP] sentence [SEP]` |
| A kettő összehasonlítása aspektusonként | RQ4 válasza |
| A modell alkalmazása a nagy Booking adatra | RQ3 válasza, nagy mintán |
| Textuális bizonyíték kiemelése (attention / token-szintű) | 10. fejezet 3. szintje |

**Módszertani figyelmeztetés, amit a dolgozatban is le kell írni:**

A HRAST 23k mondatból áll — ez *sokkal* kisebb, mint a fő adathalmazod. Ne
keverd a kettőt: a HRAST **tanítja** az ABSA modellt, a Booking adat pedig
az, amire **alkalmazod**. Ha az alkalmazás eredményeit közlöd (RQ3), mindig
írd oda, hogy ezek a modell *predikciói*, nem annotált igazság. Ez a
különbség pont az, amit a dolgozat 7. fejezetének utolsó bekezdése már
helyesen megfogalmaz (asszociáció ≠ kauzalitás).

---

### 5. fázis — Cross-dataset általánosítás (3–4 nap)

**Fejezet:** 8. / 10. | **RQ5**

| Feladat |
|---|
| A Bookingon tanított modellek futtatása az 515K adaton (ugyanaz a platform) |
| Ugyanez a TripAdvisor adaton (más platform) |
| A romlás számszerűsítése és **magyarázata** |
| A címkézési skálák közti eltérés kezelése (1–5 → 1–10 normalizálás) |

A `src/preprocessing.py` a normalizálást már megoldja — minden pontszámot
10-es skálára vetít, mielőtt címkéz. Ezt a dolgozatban meg kell említeni,
mert ez egy módszertani döntés.

---

### 6. fázis — Eredmények összeállítása és írás (2–3 hét)

Ekkorra minden ábra és táblázat megvan a `results/` mappában. Ez a fázis
a naplód rendes fejezetekké formálása.

| Dolgozat-fejezet | Miből írod meg |
|---|---|
| 4. Data Sources | 1. fázis naplója + `config.py` DATASETS regiszter |
| 6.1 Preprocessing | szűrési napló + küszöb-érzékenységi táblázat + a kétféle tisztítás indoklása |
| 6.2 Classical ML | `02_*` táblázatok |
| 6.3 Transformer | `03_*` ábrák és táblázatok |
| 7. ABSA | `04_*` + az aspektus-lexikon (függelék) |
| 8. Evaluation | confusion matrixok + hibaelemzés |
| 9. EDA | `01_*` ábrák |
| 10. Contribution | az RQ1–RQ5 válaszainak összefoglalása |

---

## Időterv

| Fázis | Idő | Kumulált |
|---|---|---|
| 0. Környezet | 0,5 nap | — |
| 1. Adat + EDA | 1 hét | 1. hét |
| 2. Klasszikus baseline | 1–1,5 hét | 2,5. hét |
| 3. BERT | 1–1,5 hét | 4. hét |
| 4. ABSA | 2 hét | 6. hét |
| 5. Cross-dataset | 3–4 nap | 6,5. hét |
| 6. Írás | 2–3 hét | **9–10. hét** |

Kb. **2,5 hónap** heti 10–15 óra ráfordítással. Számolj rá 2–3 hét
tartalékot — mindig jön valami.

---

## Három tanács, amit érdemes komolyan venni

**1. A neutral osztály lesz a fő nehézség, és ez jó hír.**
Minden modelled a semleges kategóriával fog küzdeni, mert a 6–8 közötti
pontszámú reviewk nyelvileg nem különülnek el tisztán. Ne próbáld
„megjavítani" — **elemezd**. Az, hogy pontosan megérted és megmagyarázod,
miért nehéz ez az osztály, többet ér, mint két plusz pont F1-ben.

**2. Ne hangolj a teszthalmazon.**
A teszthalmazhoz pontosan **egyszer** nyúlsz, a végén. Minden hangolás a
validációs halmazon történik. Ha többször „belenézel" és aszerint
változtatsz, a közölt szám már nem becslése a valós teljesítménynek — és
ezt egy jó bíráló észre fogja venni.

**3. A hibaelemzés a legjobban megtérülő befektetés.**
Ülj le 30 félreosztályozott reviewval, és olvasd el őket. Amit ott találsz
(irónia, tagadás, vegyes vélemény egy mondaton belül, kulturális
különbségek az értékelésben), abból lesz a dolgozat legjobb része. Metrikát
bárki tud generálni; **megérteni, hogy miért hibázik a modell, az a kutatás.**

---

## Következő lépés

1. `KORNYEZET.md` végigcsinálása (~45 perc)
2. `python tests/test_pipeline.py` — legyen zöld
3. `notebooks/01_data_loading_eda.ipynb` futtatása cellánként
4. Szólj, ha elakadsz vagy kész vagy az 1. fázissal — a 04-es ABSA
   notebookot akkor írjuk meg, amikor látod a HRAST tényleges szerkezetét
