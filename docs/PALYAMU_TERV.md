# Pályamű-terv — a szakdolgozat új szerkezete és a kód kapcsolata

> **A pályamű (a program):** egy Python-alapú adatfeldolgozó és -elemző folyamat, amely a címke nélküli
> Booking.com-hotelvéleményekhez **a szövegből, BERT-modellekkel** szentiment-címkét
> (negatív / semleges / pozitív) és **1–10-es egész pontszámot** állít elő, a címkék megbízhatóságát
> ellenőrzi, majd ezekre építve szöveg- és aspektuselemzést, valamint gyors, értelmezhető
> TF-IDF-alapú osztályozókat készít.

A kutatási kérdések (RQ1–RQ5) kikerültek: a pályamű nem kérdésekre válaszol, hanem **egy programot
mutat be** — mire való, hogyan épül fel, hogyan készült, és milyen tapasztalatokat hozott a futtatása.

---

## 1. Fejezetek a 5.2-es követelmény szerint

| Fejezet | Terjedelem | Tartalom | Forrás a kódban |
|---|---|---|---|
| **Tartalomjegyzék, tábla- és ábrajegyzék** | — | (több mint 5 ábra/tábla lesz → kell ábrajegyzék) | `results/figures`, `results/tables` |
| **1. Bevezetés** | ≤ 2 oldal | | |
| 1.1 A téma leírása és fontossága | | online vélemények szerepe a szállodaiparban; a szöveg többet mond, mint a szám | — |
| 1.2 A pályamű elkészítésének indoka | | a Booking-adat **címkézetlen**; a pontszám-vágás nem érvényes címke | `01` notebook 3. pont |
| 1.3 A pályamű elkészítésének módszere | | iteratív fejlesztés notebookokban → modulokba szervezés → tesztelés | `src/`, `tests/` |
| 1.4 Személyes indokok, köszönetnyilvánítás | | | — |
| **2. Felhasznált technológiák** | 8–10 oldal | **csak ami tényleg szerepel** | `requirements.txt` |
| 2.1 Python és az adatkezelő ökoszisztéma | | Python, pandas, NumPy, Parquet/pyarrow, Jupyter, Google Colab | |
| 2.2 Klasszikus szövegbányászat | | tisztítás, stopszó, lemmatizálás (NLTK), TF-IDF, Naive Bayes, logisztikus regresszió, lineáris SVM (scikit-learn) | `preprocessing.py`, `models_classical.py` |
| 2.3 Transzformer-modellek és a BERT | | figyelemmechanizmus, előtanítás + finomhangolás, tokenizálás, `[CLS]`, max_length | `bert_labelling.py`, `bert_training.py` |
| 2.4 PyTorch és Hugging Face Transformers | | `AutoModel…`, `Trainer`, GPU, fp16, gradiens-akkumuláció | `bert_training.py` |
| 2.5 A felhasznált kész modellek és adatok | | twitter-roberta (3 osztály), nlptown BERT (1–5 csillag), bert-base-uncased, HRAST, Booking | `config.py` → `DATASETS`, modellnevek |
| 2.6 Címke nélküli adatok kezelése | | zero-shot, transzfertanulás, önképzés (pseudo-label) — fogalmi szinten | |
| 2.7 Tematikus elemzés (2. réteg) | | többcímkés osztályozás (szigmoid, BCE, pos_weight), mondatbeágyazás (Sentence-BERT), UMAP, HDBSCAN, c-TF-IDF | `aspect_model.py`, `topic_modeling.py` |
| 2.8 Értékelési módszerek | | accuracy, makro-F1, Cohen- és Fleiss-kappa, Spearman-korreláció; többcímkés mikro/makro-F1; NPMI-koherencia, témadiverzitás, ARI-stabilitás | `evaluation.py`, `label_validation.py`, `topic_modeling.py` |
| **3. A pályamű bemutatása** | 8–10 oldal | **áttekintő ábrákkal** | |
| 3.1 Alkalmazási környezet és követelmények | | ki használná (szállodai elemző), funkcionális és nem funkcionális követelmények (reprodukálhatóság, futás 4 GB GPU-n, folytatható futás) | `config.py` |
| 3.2 Funkcionalitás | | előkészítés → címkézés (A/B/C) → végső címke → validáció → szóelemzés, kereszttáblák, aspektusok → tanítvány-modellek | `pipeline.py` lépései |
| 3.3 Architektúra | | modulábra (lásd 2. pont lent), adatfolyam-ábra, egységes kimeneti séma (`<X>_label`, `<X>_score10` …) | `src/` |
| 3.4 A megalkotás folyamata | | a fejlesztés mérföldkövei: pontszám-vágás → elvetése → BERT-címkézés; tervezési döntések (pl. két tisztítási út, mondatszintű aggregálás, cache) | régi és új kód összevetése |
| 3.5 Tapasztalatok a működtetés során | | a három megközelítés eredményei, egyezések, gold-pontosság, futási idők, szöveg ≠ pontszám példák, szóelemzés és aspektusok | `03`, `04` notebook táblái |
| **4. Összefoglalás** | 2–3 oldal | saját munka; korlátok; továbbfejlesztés | lásd 4. pont lent |
| **Hivatkozásjegyzék** | | APA 7 | lásd 5. pont lent |
| **Melléklet** | | a forráskód (`src/`), a notebookok, az aspektus-kulcsszólista, az annotálási útmutató | `CODE/` mappa |

> **Megjegyzés:** a vezetői összefoglaló legalább 20 oldal — a fenti terjedelmekkel (2 + 9 + 9 + 3) ez kb. 23 oldal.

---

## 2. Architektúra (a 3.3 fejezet ábrájához)

```
data/raw/val.csv ──► data_loading ──► preprocessing ──► [munkahalmaz]
                                                            │
                     ┌──────────────────────────────────────┼────────────────────────────┐
                     ▼                                      ▼                            ▼
           (A) zero-shot                         (B) HRAST-on tanított BERT      (C) önképzés
           2 kész modell                         bert_training.train_approach_b   A(+B) legbiztosabb
           3 osztály + 1–5★ → 1–10                mondatszint → aggregálás          címkéin tanított BERT
                     └──────────────┬───────────────────────┴────────────────────────────┘
                                    ▼
                    bert_labelling.combine_approaches  → label_final, score10_final
                                    │
            ┌───────────────────────┼─────────────────────────────┐
            ▼                       ▼                             ▼
    label_validation        text_analysis, descriptives,    models_classical
    (kappa, pontszám,       aspects (TF-IDF szóelemzés,     (TF-IDF „tanítvány”
    mezőpróba, gold)        kereszttáblák, aspektusok)        modellek)

2. INTELLIGENCIARÉTEG — tematikus kategorizálás (05 notebook)
  mondat ─► [1. réteg] szentiment (B)              → negatív / semleges / pozitív
         ─► [2/a] aspektus-osztályozó (aspect_model) → MIRŐL szól?  (HRAST-on tanított többcímkés BERT)
         ─► [2/b] altémák (topic_modeling)           → MI A GOND?   (Sentence-BERT → UMAP → HDBSCAN → c-TF-IDF)
                                                     → panasztérkép szegmensenként
```

| Modul | Feladat |
|---|---|
| `config.py` | minden beállítás egy helyen (modellnevek, küszöbök, útvonalak, színek) |
| `data_loading.py` | Booking- és HRAST-betöltés, egységes séma, parquet/CSV mentés |
| `preprocessing.py` | szűrés, kétféle tisztítás, mondatokra bontás, felosztás |
| `bert_labelling.py` | **a pályamű magja**: inferencia, 1–10 pontszám, A/B/C címkézés, pseudo-címkék, végső címke |
| `bert_training.py` | BERT finomhangolás (B és C), súlyozott veszteség, verziófüggetlen Trainer |
| `label_validation.py` | címkék ellenőrzése valódi címke nélkül |
| `text_analysis.py`, `descriptives.py`, `aspects.py` | elemzések a végső címkén |
| `models_classical.py` | TF-IDF + NB / LogReg / SVM a BERT-címkéken |
| `aspect_model.py` | **2/a réteg**: többcímkés aspektus-osztályozó a HRAST-on, küszöbhangolás, összevetés a kulcsszólistával |
| `topic_modeling.py` | **2/b réteg**: altémák felfedezése (BERTopic-elv, saját megvalósítás), minőségmérés, panasztérkép |
| `pipeline.py` | parancssori futtatás lépésenként (`python -m src.pipeline all`) |
| `tests/test_offline.py` | tesztek GPU és internet nélkül |

---

## 3. Mi változott a korábbi kódhoz képest

| Korábban | Most | Miért |
|---|---|---|
| címke = pontszám-vágás (≤7 / 8 / ≥9) | címke = BERT a **szövegből**; a vágás csak **referencia** | a konzulens jogos kifogása |
| RQ1–RQ5 cellák, kérdés-orientált notebookok | 4 notebook a program működési sorrendjében | pályamű-formátum |
| TripAdvisor, 515K betöltők (RQ5) | törölve | nincs kereszt-adathalmaz kérdés |
| HRAST csak ABSA-tervként | HRAST a (B) címkéző tanítóadata | transzfertanulás |
| 03 notebook: BERT a pontszám-címkén | 02 notebook: három címkéző megközelítés | — |
| angol kommentek | **magyar + angol** kommentek, angol változónevek | kérésed szerint |
| 2/b elemzési egység: mondat a teljes szövegből | **tagmondat**, a dicséret- és panaszmezőből külön bontva | a pozitív és negatív rész egy „mondattá” olvadt → kevert altémák |
| 2/b v1: altéma-határ 1% (19–21 altéma/aspektus, fürdőszoba ARI 0,44) | **v2:** 3% + finomság-vizsgálat (`granularity_sweep`) | túl apró, átfedő, instabil altémák |
| 2/b v2: 3% fürtméret — a fürdőszoba 18 → 2 altémára „ugrott” (86%-os óriásfürt) | **v3:** 2%-os finom fürtözés + **téma-összevonás** (max. 10 altéma, közel-duplikátumok ≥ 0,90 hasonlóságnál), `merged_from` oszlop | egy globális fürtméret nem jó minden aspektusra; „rooms are small” / „the room was quite small” két fürt volt |
| 2/b v3: középpont-alapú összevonás → 40–55%-os gyűjtőtémák (zaj: forgalom + zene + szomszédok egyben) | **v4:** átlagos kapcsolás + legfeljebb 35%-os összevont téma | „hógolyó-hatás”: az összevont középpont mindenhez hasonló |
| altémák csak az aspektusonkénti 15 000-es mintán | a mintán kívüli mondatok a legközelebbi téma-középponthoz kerülnek (`assign_to_topics`) | teljes adatnál a panasztérkép minden mondatot számoljon |
| pozitív altémák v4: vegyes listák (staff: „staff / friendly / location” 76%) | **v5:** a dicséretmezőben a vesszős felsorolás is egységekre bomlik (`split_list_items`) | a dicséretek tipikusan felsorolások, a panaszok leíró mondatok |
| panasztérkép: kék–piros skála mindkét oldalon (dicséretnél a piros = gyakoribb dicséret) | szentiment-színű skála: gyakoribb = piros (panasz) / kék (dicséret), ritkább = szürke | a piros ne jelentsen dicséretet |
| panasztérkép: nyers lift | lift + **arány-z-próba, Benjamini–Hochberg** + hatásméret (`*` = q < 0,05 ÉS \|lift − 1\| ≥ 0,2) | kis szegmensekben (n≈80) a nagy lift is lehet véletlen |
| „nothing / didn't like” szemét-téma (3,7%) | **nem-panasz szűrő** (`is_non_complaint`) a negatív altémák előtt | nem valódi panaszok |
| panasztérkép minden altémával (19–21 oszlop) | csak a **8 legnagyobb** altéma (a lift értéke nem változik) | olvashatatlan ábra |
| verzió nélküli 2/b kimenetek | `CFG.topic_version` minden 2/b fájlnévben | egy új futás ne írja felül a régit |

---

## 4. Az Összefoglalás fejezet vázlata

**Korlátok** (őszintén leírva erősítik a dolgozatot):
- a címkék **gépi („ezüst”) címkék**; a valódi pontosságot csak a kb. 400 véleményes gold minta méri;
- az (A) modellek nem hotelvéleményeken tanultak (tweetek, termékvélemények) → doménkülönbség;
- a (B) más forrásból és mondatszinten tanult → a véleményszintre aggregálás egyszerűsítés;
- a (C) a tanárai hibáit is megtanulhatja (megerősítési torzítás);
- az 1–10-es pontszám leképezése (lineáris átskálázás) tervezési döntés, nem mért skála;
- a „semleges” osztály véleményszinten gyakran **vegyes** véleményt jelent (`is_mixed`);
- csak angol nyelvű vélemények, csak a Booking `val.csv` része.

**Továbbfejlesztés:** több annotátor + annotátorok közötti egyezés; transzformer-alapú ABSA (aspektus–mondat párok);
doménadaptív előtanítás (MLM) Booking-szövegen; desztillált kisebb modell (DistilBERT) a gyorsabb futásért;
egyszerű webes felület vagy REST API a címkézőhöz; a teljes 1,6 milliós adat feldolgozása.

---

## 5. Hivatkozás-jelöltek (APA 7) — a pontos adatokat leadás előtt ellenőrizd!

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin, 76*(5), 378–382.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174.
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop on Challenges in Representation Learning*.
- Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
- Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., & Camacho-Collados, J. (2022). TimeLMs: Diachronic language models from Twitter. In *Proceedings of ACL 2022: System Demonstrations*.
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words. *Political Analysis, 16*(4), 372–403. https://doi.org/10.1093/pan/mpn018
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations* (pp. 38–45).
- Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. *Proceedings of GSCL*, 31–40.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *PAKDD 2013* (pp. 160–172). Springer.
- Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic modeling in embedding spaces. *Transactions of the ACL, 8*, 439–453.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992).
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *WSDM 2015* (pp. 399–408).
- HRAST adatcikk: „A Dataset of Hotel Reviews for Aspect-Based Sentiment Analysis and Topic Modeling” (IEEE, 2023) — szerzők és oldalszámok az IEEE Xplore oldalról.
- Booking.com Accommodation Reviews adathalmaz (Hugging Face, CC BY-SA 4.0) — az adatkártya szerinti hivatkozással.
- nlptown/bert-base-multilingual-uncased-sentiment (Hugging Face modellkártya, MIT licenc).

---

## 6. Konzulenssel egyeztetendő

1. A HRAST használata **csak a (B) címkéző tanítóadataként** rendben van-e (korábban kimaradt).
2. A gold minta mérete (javaslat: 400 vélemény) és hogy egy annotátor (te) elég-e.
3. A végső címke: többségi szavazás, vagy a gold mintán legjobb megközelítés önmagában.
4. Az 1–10-es pontszám képzése (csillagos modell várható értéke vs. polaritás-index) — melyik legyen a fő.

## 7. Munkaterv

1. `01` notebook futtatása helyben (val.csv → munkahalmaz, 20 000 véleménnyel).
2. `02` notebook Colabon: A → B (ha a HRAST megvan) → C → végső címke.
3. `03` notebook: gold minta exportja → **kézi annotálás** (kb. 4–6 óra) → értékelés.
4. Ha minden működik: `labelling_sample_size = None`, teljes futás.
5. `04` notebook: szóelemzés, kereszttáblák, aspektusok, tanítvány-modellek.
6. Írás: előbb a 3. fejezet (a program), aztán a 2. (technológiák), végül az 1. és a 4.
