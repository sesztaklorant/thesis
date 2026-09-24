# Kódútmutató — hogyan épül fel a projekt, és milyen sorrendben haladj

Ez a fájl arra való, hogy **értsd, amit futtatsz**. Nem elég, hogy lefutnak a
cellák — a védésen arra fognak kérdezni, hogy *miért* így csináltad. Ez a
dokumentum azokra a miértekre válaszol.

Három részből áll:

1. **A mentális modell** — hogyan illeszkedik össze a három réteg
2. **Az adat útja** — egy konkrét review végigkövetése a pipeline-on
3. **Sorrend** — mit olvass el, mit futtass, és mikor

---

# 1. rész — A mentális modell

## Három réteg, felülről lefelé

```
┌─────────────────────────────────────────────────────────┐
│  NOTEBOOKOK  (notebooks/)                               │
│  "Mit akarok megtudni?"                                 │
│                                                         │
│  Itt kísérletezel. Rövid cellák, sok ábra, magyarázat.  │
│  A notebook ELMESÉLI a kutatást — nem tárolja a logikát.│
└────────────────────────┬────────────────────────────────┘
                         │  importál
                         ▼
┌─────────────────────────────────────────────────────────┐
│  MODULOK  (src/)                                        │
│  "Hogyan csináljuk?"                                    │
│                                                         │
│  Az újrafelhasználható logika. Függvények, amiket több  │
│  notebook is hív. Itt van a tényleges munka.            │
└────────────────────────┬────────────────────────────────┘
                         │  olvassa
                         ▼
┌─────────────────────────────────────────────────────────┐
│  KONFIGURÁCIÓ  (src/config.py)                          │
│  "Milyen beállításokkal?"                               │
│                                                         │
│  Minden szám, küszöb, útvonal. EGY helyen.              │
└─────────────────────────────────────────────────────────┘
```

### Miért van ez így szétválasztva?

**Ha minden a notebookban lenne**, három baj történne:

1. **Nem tudnád megismételni.** A notebook cellái tetszőleges sorrendben
   futtathatók. Ha a 12. cellát kétszer futtatod, a 7-et meg egyszer sem,
   más eredményt kapsz — és fogalmad sem lesz, miért.
2. **Duplikálnád a kódot.** A tisztítás kell az EDA-hoz, a TF-IDF-hez és a
   BERT-hez is. Három másolat három helyen = három hely, ahol elromolhat.
3. **A bíráló nem tudná átnézni.** Egy 200 cellás notebookban nem látszik a
   módszertan. Egy `preprocessing.py`-ban igen.

**Ha minden a modulokban lenne**, elveszne a kutatás elmesélése — a notebook
pont attól hasznos, hogy látod benne a köztes eredményeket és az ábrákat.

A megosztás szabálya egyszerű: **ami kétszer kell, az modulba megy; ami egyszeri
kísérlet vagy magyarázat, az a notebookban marad.**

---

## Mappastruktúra — mi hol lakik

```
CODE/
│
├── src/                    ← A KÓD. Ide nyúlsz, ha a MÓDSZEREN változtatsz.
│   ├── config.py           · minden beállítás
│   ├── data_loading.py     · adat beolvasása egységes formára
│   ├── preprocessing.py    · tisztítás, címkézés, felosztás
│   ├── models_classical.py · TF-IDF + NB/LogReg/SVM
│   ├── evaluation.py       · metrikák, ábrák, hibaelemzés
│   ├── aspects.py          · ABSA: mondatbontás, aspektusok
│   └── __init__.py         · set_seed(), naplózás
│
├── notebooks/              ← A KUTATÁS. Itt dolgozol nap mint nap.
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_classical_baseline.ipynb
│   └── 03_bert_colab.ipynb
│
├── data/                   ← ADAT. Git-ből kizárva (méret + licenc).
│   ├── raw/                · érintetlen letöltés — SOHA ne írd felül
│   ├── interim/            · félkész (most nem használjuk)
│   └── processed/          · modellezésre kész, splitelt parquet
│
├── results/                ← AMI A DOLGOZATBA MEGY.
│   ├── figures/            · 300 DPI ábrák
│   ├── tables/             · CSV táblázatok
│   └── models/             · mentett modellek (Git-ből kizárva)
│
├── docs/
│   ├── naplo.md            · kutatási napló — EBBŐL LESZ A DOLGOZAT SZÖVEGE
│   └── KOD_UTMUTATO.md     · ez a fájl
│
├── tests/test_pipeline.py  ← FÜSTTESZT. Ha valami elromlik, ezzel kezdd.
│
├── MUNKATERV.md            · fázisok, ütemterv, RQ-megfeleltetés
├── KORNYEZET.md            · a környezet dokumentációja
├── requirements.txt        · a csomagok listája
└── .gitignore              · mi ne kerüljön a repóba
```

### A három mappa, amit értened kell

**`data/raw/` — az érinthetetlen.** Ide kerül a letöltött adat, és **soha nem
írod felül**. Ha elrontasz egy tisztítást, ide tudsz visszanyúlni anélkül, hogy
újra letöltenél 1,6 millió sort. Ez általános adattudományi konvenció, nem az én
ötletem — „raw data is immutable".

**`data/processed/` — a szerződés a notebookok között.** A 01-es notebook ide
írja a `booking_train.parquet`, `booking_val.parquet`, `booking_test.parquet`
fájlokat. A 02-es és 03-as innen olvas. Így nem kell újra előfeldolgozni minden
alkalommal, **és mindhárom modell PONTOSAN ugyanazon a felosztáson dolgozik** —
enélkül az RQ1 összehasonlítás értelmetlen lenne.

**`results/` — a dolgozat nyersanyaga.** Minden ábra 300 DPI-vel mentődik (nyomdai
minőség), minden táblázat CSV-be. Amikor írni kezded a dolgozatot, innen
másolsz — nem futtatod újra a notebookokat.

---

## Modulonként: mit csinál és miért

### `config.py` — a szerződés

Két dataclass és egy szótár:

- **`PATHS`** — minden elérési út. A `PROJECT_ROOT` a fájl helyéből számolódik,
  ezért mindegy, honnan futtatod (notebookból, szkriptből, tesztből).
- **`CFG`** — minden beállítás: seed, küszöbök, split arányok, TF-IDF paraméterek,
  BERT hiperparaméterek, aspektuslista.
- **`DATASETS`** — a négy adathalmaz regisztere: honnan jön, mi a licenc, mi a
  szerepe, mik az oszlopnevei.

**Miért fontos ez?** Mert a védésen valaki megkérdezi: *„milyen küszöbbel képezte a
semleges osztályt?"* — és te egyetlen fájlra tudsz mutatni, nem tizenöt notebookcellát
kezdesz keresgélni. A `DATASETS` regiszterből pedig a dolgozat 4. fejezete
(Data Sources) közvetlenül megírható.

A `CFG.to_json()` menti a beállításokat az adat mellé — így minden eredményhez
tudni fogod, pontosan milyen konfigurációval készült.

**Amit itt módosítani fogsz:** `dev_sample_size` (fejlesztés közben 50 000, a
végleges futtatáshoz `None`), és esetleg a `neg_threshold` / `pos_threshold`,
ha az érzékenységi vizsgálat mást javasol.

---

### `data_loading.py` — négy adathalmaz, egy forma

A három hotel-adathalmaz teljesen más szerkezetű. A Booking `review_positive` és
`review_negative` mezőket használ 1–10-es pontszámmal, a TripAdvisor egyetlen
`Review` mezőt 1–5 csillaggal, a HRAST mondatszintű annotációkat.

Ez a modul mindegyikből **ugyanazt a táblát** állítja elő:

```
review_id | text | score | score_scale_max | dataset | (extra oszlopok)
```

**Miért éri meg ez a munka?** Mert az RQ5 (cross-dataset általánosítás) így
triviális lesz:

```python
train_source = load_dataset_by_name("booking_hf", n_rows=50_000)
external     = load_dataset_by_name("tripadvisor")
# ugyanaz a modell, ugyanaz a kiértékelő kód, más adat
```

Enélkül minden adathalmazhoz külön kódot írnál, és soha nem lennél biztos benne,
hogy tényleg ugyanazt méred.

**Két részlet, amit érdemes megnézni a kódban:**

1. **Gyorsítótárazás.** Az első letöltés után parquet fájlba ment. Másodszorra
   másodpercek alatt betölt. Ezért van a `use_cache=True` paraméter.
2. **A `_clean_part` függvény.** A Booking adatban a hiányzó rész gyakran nem
   üres, hanem a „Nothing" szó szerepel benne. Ha ezt nem szűrnéd, a modell
   megtanulná, hogy a „nothing" szó pozitív jelzés — ami nyilvánvalóan hamis
   összefüggés. **Ez pont az a fajta adatminőségi megfigyelés, ami a dolgozat
   6.1 fejezetébe való.**

---

### `preprocessing.py` — a legfontosabb modul

Itt van a legtöbb módszertani döntés. Négy dolgot csinál:

#### (a) Kétféle tisztítás — és ez szándékos

```python
clean_text_minimal(text)   # BERT-hez
clean_text_classic(text)   # TF-IDF-hez
```

**Miért nem ugyanaz?**

A TF-IDF egy „zsák-a-szavak" modell: nem érti a szórendet, csak azt látja, mely
szavak fordulnak elő és milyen gyakran. Neki segít az agresszív normalizálás —
kisbetűsítés, írásjelek eltávolítása, stopword-szűrés, lemmatizálás. Kevesebb zaj,
kisebb szókincs, jobb általánosítás.

A BERT viszont a **kontextusból** tanul. Ha eltávolítod a stopwordöket, a
`"not clean"` kifejezésből `"clean"` lesz — vagyis az **ellenkezőjét** tanítod
neki. Ezért kapja a BERT a minimálisan tisztított szöveget: csak a HTML és az
URL megy ki, az írásjelek, nagybetűk és tagadószavak maradnak.

Ez nem apróság: **ez egy önálló bekezdés a dolgozat 6.1 fejezetében**, és
közvetlenül összefügg az RQ2-vel.

A `clean_text_classic` minden lépése külön kikapcsolható (`lowercase`,
`remove_punct`, `remove_stopwords`, `lemmatize`, `keep_negations`) — pontosan
azért, hogy ablációs vizsgálatot tudj futtatni.

#### (b) Címkézés pontszámból

```python
score <= 6  ->  negative
6 < score < 8  ->  neutral
score >= 8  ->  positive
```

**Ez a projekt legvitathatóbb döntése, és tudatosan az.** Az adatban nincs kész
sentiment-címke, csak vendégpontszám. A 6/8 vágás a szakirodalomban gyakori, de
önkényes.

Két dolog védi meg:

1. **Minden skála 10-esre normalizálódik** a címkézés előtt. A TripAdvisor 1–5-ös
   csillagai `score * 2` értékké válnak. Enélkül az RQ5 összehasonlítás hamis
   lenne, mert más szabály vonatkozna a két adathalmazra.
2. **Érzékenységi vizsgálat** a 01-es notebookban: megnézed, hogyan változik az
   osztályeloszlás más küszöbökkel, és ezt táblázatban közlöd.

#### (c) Szűrés naplóval

A `filter_reviews()` nem csak szűr, hanem **kiírja, hány sort veszítettél melyik
lépésben**. Ezt a táblázatot tedd be a dolgozatba — a bírálók szeretik látni az
adattisztítás számszerű hatását.

Szűrési lépések: túl rövid szöveg, extrém hosszú, duplikátum, nem angol.

> A nyelvfelismerés heurisztika: ha a karakterek 95%-a ASCII, valószínűleg angol.
> Ez olcsóbb, mint a `langdetect` 1,6 millió soron, és a hotel-adatokban jól
> működik. Ha a dolgozatban pontosabbat akarsz állítani, futtasd le a
> `langdetect`-et egy 1000-es mintán, és közöld, hogy egyezik-e.

#### (d) 60/20/20 felosztás — rétegzetten

```python
train, val, test = train_val_test_split(df)
```

Két lépésben oszt: előbb leválasztja a tanítót (60%), majd a maradékot felezi.

**A `stratify=True` itt nem opció, hanem követelmény.** Az adatod erősen
kiegyensúlyozatlan (a pozitív osztály jellemzően 60–70%). Ha véletlenszerűen
osztanál, a teszthalmazba a véletlen miatt aránytalanul kevés `neutral` kerülhetne,
és a mért teljesítmény nem lenne megbízható.

**Mire való a három halmaz — ezt tudnod kell a védésen:**

| Halmaz | Mire | Hányszor nyúlsz hozzá |
|---|---|---|
| **train** (60%) | modelltanítás | ahányszor akarsz |
| **validation** (20%) | modellválasztás, hiperparaméter-hangolás | ahányszor akarsz |
| **test** (20%) | a végleges teljesítmény mérése | **PONTOSAN EGYSZER** |

Ha a teszthalmazba „belenézel", és aszerint hangolsz, akkor a teszt már nem
független — az eredményed optimista lesz, és a dolgozatban közölt szám nem
becslése a valós teljesítménynek. Ez a leggyakrabban keresett módszertani hiba.

---

### `models_classical.py` — TF-IDF és a három modell

#### A Pipeline, ami megelőzi az adatszivárgást

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf",   LogisticRegression(...)),
])
```

**Ez a projekt legfontosabb technikai részlete.** Ha külön illesztenéd a TF-IDF-et
az *egész* adathalmazon, majd külön tanítanád a modellt a tanító részen, akkor a
szókincs és az IDF-súlyok tartalmaznák a teszthalmaz információját is. Ez
**adatszivárgás** (data leakage), és az eredményed hamisan jó lenne.

A Pipeline-nal a `fit()` csak a tanító adatot látja, a `predict()` pedig már
csak transzformál a megtanult szókinccsel. A teszthalmazban lévő ismeretlen szavak
egyszerűen kimaradnak — ahogy a valóságban is történne.

#### A TF-IDF paraméterek jelentése

| Paraméter | Érték | Mit csinál |
|---|---|---|
| `ngram_range` | (1, 2) | egy- és kétszavas kifejezések. A `"not clean"` bigram sentimentnél aranyat ér |
| `min_df` | 3 | a 3-nál kevesebb dokumentumban előforduló szó zaj (elgépelés, egyedi név) |
| `max_df` | 0.9 | a reviewk 90%-ában előforduló szó (`"hotel"`) nem megkülönböztető |
| `sublinear_tf` | True | `1 + log(tf)` a nyers gyakoriság helyett — tompítja az ismétlődéseket |
| `max_features` | 50 000 | a szókincs felső korlátja |

#### `class_weight="balanced"`

A ritkább osztály (a `neutral`) hibái nagyobb súlyt kapnak. Kiegyensúlyozatlan
adaton ez jelentősen javítja a macro-F1-et. A Naive Bayes nem támogatja, ott
marad az alapértelmezés — ezt a dolgozatban érdemes megemlíteni.

#### `preprocessing_ablation()` — az RQ2 közvetlen válasza

Nyolc variánst futtat ugyanazzal a modellel, különböző előfeldolgozással és
TF-IDF beállítással, és táblázatot ad vissza. Ez a táblázat egy az egyben mehet a
dolgozatba.

**Amit figyelj benne:** a `"tisztítás, tagadószavak eldobva"` variáns.
Valószínűleg mérhetően rosszabb lesz — és ez egy erős, jól bemutatható
megfigyelés arról, hogy miért korlátozott a zsák-a-szavak megközelítés.

#### `top_features_per_class()` — értelmezhetőség

Kiszedi a lineáris modell legnagyobb súlyú jellemzőit osztályonként. A negatív
osztálynál ott lesznek a `dirty`, `noisy`, `smell`, `rude`, `overpriced` szavak.

**Ez köti össze a klasszikus modellt az ABSA résszel:** a modell „magától"
megtalálja ugyanazokat az aspektusokat, amiket te a kulcsszólistával definiáltál.
A két megközelítés validálja egymást — ez egy jó érv a dolgozatban.

---

### `evaluation.py` — közös mérce

**Ez a modul teszi lehetővé az RQ1-et.** Mind a klasszikus modelleket, mind a
BERT-et **ugyanezekkel a függvényekkel** értékeled ki. Ha külön metrikakódot
írnál a kettőhöz, soha nem lehetnél biztos benne, hogy ugyanazt méred.

#### Macro-F1 vs. weighted-F1 — a legfontosabb fogalom

| Metrika | Mit csinál | Mikor félrevezető |
|---|---|---|
| **accuracy** | a helyes tippek aránya | ha a pozitív osztály 70%, egy „mindent pozitívnak mondok" modell is 70%-ot kap |
| **weighted-F1** | osztályonkénti F1, gyakoriság szerint súlyozva | ugyanez a probléma, enyhébben |
| **macro-F1** | osztályonkénti F1, **egyenlő súllyal** | — |

**A fő metrikád a macro-F1 legyen, és ezt írd is le a dolgozatban.** Ez az
egyetlen, ami megmutatja, hogy a modell tényleg megtanulta-e a nehéz `neutral`
osztályt, vagy csak a többség felé tippel.

A kettő **különbsége** is árulkodó: minél nagyobb a rés, annál rosszabbul kezeli
a modell a kisebbségi osztályt.

#### `plot_confusion_matrix()` — soronként normalizálva

A sorok a **valós** osztályok, az oszlopok a **predikáltak**. A normalizálás
soronként történik, tehát minden sor azt mutatja: „az adott valós osztály hány
százalékát tippelte a modell melyik osztálynak?"

**Amit látni fogsz:** a `neutral` sor szétfolyik a másik kettőbe. Ez lesz a
legérdekesebb megfigyelésed, és a hibaelemzés fejezet alapja.

#### `error_analysis()` — a legjobban megtérülő befektetés

Kigyűjti a félreosztályozott reviewkat hibatípusonként. **Olvasd el kézzel 20–30
példát.** Metrikát bárki tud generálni; megérteni, hogy *miért* hibázik a modell,
az a kutatás.

Tipikus felfedezések: irónia, tagadás, egy reviewn belüli vegyes vélemény,
kulturális különbségek az értékelésben (valaki 7-et ad ugyanarra, amire más 9-et).

---

### `aspects.py` — ABSA baseline

Négy lépés:

1. **`explode_to_sentences()`** — review-szintű táblából mondat-szintű tábla.
   Ez azért kell, mert egy review több, egymásnak ellentmondó véleményt
   tartalmazhat — pont ez a dolgozatod kiinduló példája („great location, but the
   room was extremely hot").
2. **`detect_aspects()`** — kulcsszólistákkal megnézi, mely aspektusok szerepelnek.
   A lexikon (`ASPECT_KEYWORDS`) **a dolgozat függelékébe való**, mert ez egy
   módszertani döntés.
3. **`aspect_sentiment_table()`** — az **RQ3 közvetlen válasza**. Aspektusonként
   megmutatja a sentiment megoszlását és a `neg_lift` értéket:
   - `neg_lift > 1` → az aspektus **panaszhajtó** (az átlagnál gyakrabban negatív)
   - `neg_lift < 1` → **elégedettséghajtó**
4. **`extract_evidence()`** — az aspektus-kulcsszó körüli szókörnyezet. Ez a
   dolgozat 10. fejezetének harmadik szintje („textuális bizonyíték").

**Fontos korlát, amit a dolgozatban is le kell írni:** itt a **review-szintű**
sentimentet vetíted rá a mondatokra, ami közelítés. Egy pozitív összpontszámú
review is tartalmazhat negatív mondatot. A pontos mondat-szintű címkékhez kell a
HRAST (4. fázis).

**Amit keresni kell:** hol bukik el a kulcsszavas módszer. Tagadás („the room was
**not** clean"), irónia, implicit aspektus („we could not sleep" = zaj, kulcsszó
nélkül), többszörös aspektus egy mondatban. **Ez a lista lesz az indoklásod arra,
miért kell transformer-alapú ABSA** — és egy önálló, értékes alfejezet.

---

### `__init__.py` — `set_seed()`

```python
set_seed()   # a notebook ELSŐ cellájában
```

Egyetlen hívással rögzíti a Python, a NumPy és a PyTorch véletlenszám-generátorát.

**Enélkül minden futtatásnál kicsit más eredményt kapsz**, és a dolgozatba írt
számokat nem tudod reprodukálni. A védésen ez kellemetlen kérdéseket szül.

---

# 2. rész — Az adat útja

Kövessünk végig **egy konkrét reviewt** a teljes pipeline-on. Ha ezt érted,
érted az egész projektet.

## Kiindulás: a nyers sor a Hugging Face-ről

```
review_title:     "Great spot, shame about the room"
review_positive:  "Amazing location, right by the metro."
review_negative:  "The room was extremely hot, no air conditioning."
guest_score:      6.5
```

## 1. `load_booking_hf()` → kanonikus forma

Összefűzi a három szövegmezőt, és egységes oszlopneveket ad:

```
review_id:         "booking_hf_4711"
text:              "Great spot, shame about the room. Amazing location,
                    right by the metro. The room was extremely hot, no
                    air conditioning."
score:             6.5
score_scale_max:   10.0
dataset:           "booking_hf"
```

→ elmentve: `data/raw/booking_hf.parquet`

## 2. `filter_reviews()` → átmegy a szűrőn

- hossz > 10 karakter ✅
- hossz < 5000 karakter ✅
- nem duplikátum ✅
- 95%+ ASCII → angol ✅

Marad.

## 3. `add_sentiment_labels()` → címke

```
6.5 * (10/10) = 6.5
6.0 < 6.5 < 8.0   →   sentiment = "neutral"
```

**Figyeld meg, milyen nehéz eset ez.** A szöveg tartalmaz egy erősen pozitív és
egy erősen negatív részt. A vendég 6,5-öt adott, ezért „semleges" a címkéje — de
nyelvileg ez nem semleges szöveg, hanem két ellentétes vélemény egymás mellett.

**Ez a mondat magyarázza meg, miért lesz a `neutral` osztály a legnehezebb**, és
ez az a megfigyelés, amit a hibaelemzés fejezetbe írsz.

## 4. Kétféle tisztítás

```
text_bert:   "Great spot, shame about the room. Amazing location, right by
              the metro. The room was extremely hot, no air conditioning."
             (szinte változatlan — a BERT a kontextusból tanul)

text_clean:  "great spot shame room amazing location right metro room
              extremely hot no air conditioning"
             (kisbetűs, írásjelek nélkül, stopwordök nélkül, lemmatizálva —
              DE a "no" megmaradt, mert keep_negations=True)
```

→ `data/processed/booking_*.parquet`

## 5. Felosztás

A rétegzett split miatt ez a sor valamelyik halmazba kerül, és az arányok
mindhárom halmazban megmaradnak.

## 6/a. TF-IDF útvonal (02-es notebook)

A `text_clean` számvektorrá alakul — a `no air conditioning` trigram, az
`extremely hot` bigram mind külön jellemző lesz. A modell megtanulja, hogy ezek
a negatív osztály felé húznak. Aztán tippel — valószínűleg `negative`-ot, mert a
negatív jelek erősebbek. **Hibázik**, és bekerül a hibaelemzésbe.

## 6/b. BERT útvonal (03-as notebook)

A `text_bert` tokenekre bomlik, és a BERT az egész mondatot **kontextusban**
dolgozza fel. Érti, hogy a `"shame about"` megszorítás, és hogy a review két
részből áll. Jó eséllyel közelebb kerül a `neutral`-hoz — **és pont ez lesz az
RQ1 válaszának a lényege.**

## 7. ABSA útvonal (`aspects.py`)

Mondatokra bomlik:

```
"Great spot, shame about the room."              → aspects: [room]
"Amazing location, right by the metro."          → aspects: [location]
"The room was extremely hot, no air conditioning." → aspects: [room, facilities]
```

Bizonyíték-kiemelés:

```
room → "The room was extremely hot, no"     [TAGADÁS!]
```

**Itt látszik a kulcsszavas módszer korlátja:** megtalálta a `room` aspektust és
a környezetét, de nem tudja, hogy a `no air conditioning` **negatív** — csak azt
jelzi, hogy van benne tagadás. A transformer ABSA ezt fogja megoldani.

---

# 3. rész — Sorrend

## A. Olvasási sorrend — mielőtt bármit futtatnál

Szánj rá **egy-másfél órát**. Sokkal gyorsabban fogsz haladni utána.

| # | Fájl | Mire figyelj | Idő |
|---|---|---|---|
| 1 | `src/config.py` | Mik a beállítások? Melyik számot mi indokolja? | 15 perc |
| 2 | `src/preprocessing.py` | A kétféle tisztítás. A címkézés. A split. | 25 perc |
| 3 | `src/evaluation.py` | Macro vs. weighted F1. | 15 perc |
| 4 | `src/models_classical.py` | A Pipeline és az adatszivárgás. | 15 perc |
| 5 | `src/aspects.py` | A lexikon és a `neg_lift` logika. | 15 perc |
| 6 | `src/data_loading.py` | Elég átfutni — ez a legkevésbé érdekes. | 10 perc |

**Ne próbáld megjegyezni.** A cél az, hogy tudd, *hol keresd*, amikor kell.

Minden fájl tetején van egy magyarázó blokk, és a fontos döntéseknél kommentek.
Ha valami nem világos, kérdezz rá — ez pont az a fajta kérdés, amire érdemes
választ kapni, mielőtt továbbmész.

## B. Futtatási sorrend

### 1. fázis — Adat és EDA (kb. 1 hét)

Notebook: `01_data_loading_eda.ipynb`

```
[1] Környezet, set_seed()            ← mindig ezzel kezdj
[2] Adat betöltése (50k minta)       ← első futtatás pár perc
[3] Adatminőség: hiányzó, duplikátum, hosszeloszlás
[4] Címkézés + KÜSZÖB-ÉRZÉKENYSÉGI VIZSGÁLAT
[5] Szűrés és tisztítás              ← lassú (lemmatizálás)
[6] EDA-ábrák                        ← 3 ábra a dolgozatba
[7] Aspektus-előzetes                ← RQ3 első válasza
[8] 60/20/20 felosztás és mentés     ← ez táplálja a 02-est és 03-ast
```

**Állj meg a 4. lépés után**, és nézd meg az osztályeloszlást. Ha a `neutral`
osztály 10% alatt van, érdemes elgondolkodni a küszöbökön — szólj, és
megbeszéljük.

**A fázis végén írd meg a naplót** (`docs/naplo.md`): miért ez az adathalmaz, mi
a licenc, miért 6/8 a küszöb, mit dobott ki a szűrés, mit láttál az EDA-ban.

### 2. fázis — Klasszikus baseline (1–1,5 hét)

Notebook: `02_classical_baseline.ipynb`

```
[1-2] Betöltés + három modell tanítása   ← percek
[3]   A legjobb a TESZT halmazon         ← EGYSZER!
[4]   ABLÁCIÓ (RQ2)                      ← ez a leglassabb, 8 variáns
[5]   Top jellemzők
[6]   HIBAELEMZÉS                        ← olvass el 20-30 példát kézzel
[7]   Cross-dataset (RQ5 első fele)
```

**Jegyezd fel a legjobb macro-F1-et.** Ez lesz a BERT viszonyítási alapja.

### 3. fázis — BERT (1–1,5 hét)

Notebook: `03_bert_colab.ipynb` — a saját GPU-don, a 4 GB-os beállításokkal
(`BATCH_SIZE=8`, `GRAD_ACCUM=4`, `fp16=True`).

Első futtatásnál **kisebb mintán** próbáld (5–10 ezer sor), hogy lásd, végigmegy-e
memóriahiba nélkül. Ha igen, jöhet a teljes.

### 4. fázis — ABSA (2 hét)

A `04_absa_transformer.ipynb`-t **együtt írjuk meg**, amikor letöltötted a
HRAST-ot, és látjuk a tényleges oszlopszerkezetét.

### 5–6. fázis — Cross-dataset, majd írás

Részletek a `MUNKATERV.md`-ben.

---

## C. Hogyan érts meg egy cellát, ami nem világos

Négy gyakorlati fogás, amit használj bátran:

**1. Nézd meg, mi jött ki.** Szúrj be egy cellát bárhova:

```python
df.head(3)              # az első 3 sor
df.shape                # (sorok, oszlopok)
df.dtypes               # milyen típusú minden oszlop
df["sentiment"].value_counts()
```

**2. Ugorj a definícióhoz.** VS Code-ban `Ctrl+kattintás` bármelyik
függvénynéven → odaugrik a forráskódhoz. `Alt+←` visszahoz.

**3. Futtass egy sort külön.** Ha egy hosszú láncot nem értesz, bontsd szét:

```python
# eredeti:
df = pp.build_dataset(df_raw, classic_clean=True)

# szétbontva:
tmp = pp.filter_reviews(df_raw)
print(tmp.shape)
tmp = pp.add_sentiment_labels(tmp)
print(tmp["sentiment"].value_counts())
```

**4. Kérdezz rá.** Másold be a cellát és a hibaüzenetet — végigvesszük.

---

## D. Amit a védésre tudnod kell

Ezekre a kérdésekre legyen kész válaszod. Ha most még nincs, a naplód végére
lesz:

1. Miért ezt az adathalmazt választottad, és mi a licenc?
2. Hogyan képezted a három osztályt, és miért éppen azokkal a küszöbökkel?
3. Miért más az előfeldolgozás a TF-IDF-hez és a BERT-hez?
4. Miért a macro-F1 a fő metrikád, és nem az accuracy?
5. Hogyan előzted meg az adatszivárgást?
6. Hányszor használtad a teszthalmazt?
7. Miért nehéz a `neutral` osztály?
8. Hol bukik el a kulcsszavas aspektusdetektálás, és mit old meg helyette a
   transformer?
9. Mennyivel jobb a BERT, és milyen számítási áron?

**Mindegyikre a naplódból fogsz tudni válaszolni** — ha vezeted. Ez az egyetlen
dolog, amit tényleg minden fázis után csinálj meg.

---

## Az első lépés most

```powershell
cd C:\Users\seszt\Documents\BCE\7_felev\Szakdolgozat\THESIS\CODE
.venv\Scripts\Activate.ps1
code .
```

VS Code-ban nyisd meg a `src/config.py`-t, és olvasd végig. Utána a
`notebooks/01_data_loading_eda.ipynb`-t — de csak az első két cellát futtasd,
és nézd meg, mi jött be.

Az első valódi akadály a 2. cella lesz: a Booking.com adat letöltése a Hugging
Face-ről. Ha hibát dob, másold be.
