# Környezet beállítása — lépésről lépésre

Ez az útmutató feltételezi, hogy Windowson vagy macOS-en dolgozol, és most
állítasz be először Python fejlesztői környezetet. Menj végig rajta sorban;
minden lépés végén van egy **ellenőrzés**, amivel látod, hogy sikerült-e.

**Összesen kb. 30–45 perc**, plusz a letöltési idők.

---

## Áttekintés: mi hol fut?

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  A LAPTOPOD                  │     │  GOOGLE COLAB (ingyenes)     │
│                              │     │                              │
│  • adatletöltés, tisztítás   │     │  • BERT finomhangolás        │
│  • feltáró elemzés (EDA)     │ ──▶ │  • transformer ABSA          │
│  • TF-IDF + NB/LogReg/SVM    │     │                              │
│  • ábrák, táblázatok         │ ◀── │  T4 GPU, ~15-25 perc/futás   │
│  • a dolgozat írása          │     │                              │
│                              │     │  (laptopon CPU-n: 8-15 ÓRA)  │
└──────────────────────────────┘     └──────────────────────────────┘
         a parquet fájlokat feltöltöd ──▶
         a betanított modellt letöltöd ◀──
```

A munka **80%-a a laptopon** történik. A Colab csak a GPU-t igénylő
tanításhoz kell.

---

## 1. lépés — Python telepítése

### Windows

1. Töltsd le: <https://www.python.org/downloads/> → **Python 3.11.x**
   (ne a legújabb 3.13/3.14-et — a PyTorch és a transformers néha
   hetekkel-hónapokkal lemaradva támogatja az új verziókat)
2. A telepítőben **PIPÁLD BE** legalul: **„Add python.exe to PATH"**
   ← ez a leggyakoribb hiba forrása, ne hagyd ki
3. Install Now

### macOS

```bash
# Homebrew-val (ha nincs: https://brew.sh)
brew install python@3.11
```

### Ellenőrzés

Nyiss egy **új** terminált (Windows: PowerShell, macOS: Terminal):

```bash
python --version      # Windowson
python3 --version     # macOS-en
```

Ezt kell látnod: `Python 3.11.x`
Ha „nem található" hibát kapsz Windowson, a PATH pipa maradt ki → telepítsd újra.

---

## 2. lépés — VS Code telepítése

Kódszerkesztő. A notebookokat közvetlenül benne tudod futtatni, ami
kényelmesebb, mint a böngészős Jupyter.

1. <https://code.visualstudio.com/> → Download → telepítés
2. Indítsd el, bal oldalon a **Extensions** ikon (négy kocka)
3. Telepítsd ezt a kettőt:
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)

---

## 3. lépés — Git telepítése és a projekt verziókövetése

**Miért kell ez a szakdolgozathoz?** Mert egyszer biztosan elrontasz valamit,
és vissza akarsz térni a tegnapi működő állapothoz. Plusz a témavezetőnek
meg tudod mutatni, hogyan haladt a munka — ez komoly pluszpont.

1. <https://git-scm.com/downloads> → telepítés (minden alapbeállítás jó)
2. Add meg, ki vagy:

```bash
git config --global user.name "Sesztak Lorant"
git config --global user.email "sesztaklorant@gmail.com"
```

### Ellenőrzés

```bash
git --version     # git version 2.4x.x
```

---

## 4. lépés — A projekt kicsomagolása és a virtuális környezet

### 4.1 Hova tedd a projektet?

Csomagold ki a `hotel-absa` mappát egy **ékezet és szóköz nélküli** útvonalra.
Ez komolyan számít — a Python egyes könyvtárai megbotlanak az ékezetes
útvonalakon:

- Jó: `C:\szakdolgozat\hotel-absa` vagy `~/szakdolgozat/hotel-absa`
- Rossz: `C:\Users\Lóránt\Dokumentumok\Szakdolgozat 2026\...`

### 4.2 Virtuális környezet (venv)

**Mi ez?** Egy elkülönített Python-példány, csak ehhez a projekthez. Így a
projekt csomagjai nem ütköznek más projektek csomagjaival, és a
`requirements.txt` pontosan leírja, mi kell a futtatáshoz — ez a
reprodukálhatóság alapja.

Nyisd meg a projektmappát terminálban:

```bash
# Windows (PowerShell)
cd C:\szakdolgozat\hotel-absa
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
cd ~/szakdolgozat/hotel-absa
python3 -m venv .venv
source .venv/bin/activate
```

> **Windows hiba:** „a szkriptek futtatása le van tiltva"?
> Futtasd ezt egyszer, majd próbáld újra az aktiválást:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### Ellenőrzés

A terminál sor elején meg kell jelennie: `(.venv)`
Ha ott van, jó helyen vagy. **Minden alkalommal aktiválnod kell**, amikor új
terminált nyitsz a projekthez.

### 4.3 Csomagok telepítése

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Ez 3–6 percig tart, és kb. 800 MB-ot tölt le.

### 4.4 NLTK nyelvi erőforrások

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

### Ellenőrzés — füstteszt

```bash
python tests/test_pipeline.py
```

Ennek kell a végén kiírnia:

```
==================================================================
 MINDEN TESZT SIKERES — a környezeted készen áll.
==================================================================
```

**Ha ez lefut, a környezeted kész.** A teszt szintetikus adaton futtatja
végig a teljes pipeline-t: tisztítás → címkézés → felosztás → három modell
betanítása → kiértékelés → ABSA.

### 4.5 Git repó inicializálása

```bash
git init
git add .
git commit -m "Kezdeti projektvaz: config, src modulok, notebookok"
```

> A `.gitignore` gondoskodik róla, hogy az adatok és a betanított modellek
> **ne** kerüljenek a repóba (túl nagyok, és licencproblémát okozhatnak).

Ha a témavezetőddel is meg akarod osztani: csinálj egy **privát** repót
GitHubon, és:

```bash
git remote add origin https://github.com/FELHASZNALONEVED/hotel-absa.git
git branch -M main
git push -u origin main
```

---

## 5. lépés — PyTorch (csak ha helyben is akarsz BERT-tel próbálkozni)

**Ezt nyugodtan kihagyhatod**, ha a BERT-et a Colabon futtatod — ami az
ajánlott út. Csak akkor telepítsd, ha helyben is szeretnél kísérletezni
kis mintán.

```bash
# CPU-s változat (kisebb, ~200 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate evaluate
```

Ha van NVIDIA GPU-d, a megfelelő CUDA-s parancsot a
<https://pytorch.org/get-started/locally/> oldalon állítsd össze.

---

## 6. lépés — Kaggle API (az 515K / TripAdvisor / HRAST adathoz)

A Booking.com adathalmaz Hugging Face-ről jön, ahhoz nem kell fiók.
A többihez Kaggle-fiók kell.

1. Regisztrálj: <https://www.kaggle.com>
2. Jobb felül a profilképed → **Settings** → **API** → **Create New Token**
3. Letöltődik egy `kaggle.json` fájl. Tedd ide:
   - Windows: `C:\Users\<neved>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`

```bash
# macOS/Linux — jogosultság beállítása (különben figyelmeztet)
chmod 600 ~/.kaggle/kaggle.json
```

> **Ha ez bonyolultnak tűnik:** teljesen rendben van kézzel letölteni a
> CSV-ket a böngészőből, és bemásolni a `data/raw/` mappába. A kód mindkét
> utat megtalálja.

---

## 7. lépés — Google Colab

1. Menj a <https://colab.research.google.com> oldalra (Google-fiók kell)
2. **File → Upload notebook** → válaszd a `notebooks/03_bert_colab.ipynb`-t
3. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → Save
4. Az első cella kiírja, hogy tényleg GPU-n futsz-e

### Colab tudnivalók

- Az ingyenes verzió ~90 perc inaktivitás után lecsatlakozik → **a fájlok elvesznek**
- Napi GPU-kvóta van (kb. 3–4 óra); ha elfogy, várj pár órát
- **Mindig töltsd le, amit előállítasz** (a notebook utolsó cellája ezt megteszi)
- Ha sokat fogsz futtatni, a Colab Pro (~10 EUR/hó) megéri — de nem kötelező

### Fájlmozgatás laptop ↔ Colab

**Egyszerű út:** a bal oldali mappa ikonra kattintva fel/le tudsz tölteni fájlokat.

**Kényelmesebb út (ajánlott):** Google Drive csatolása. Készíts egy
`szakdolgozat/data/processed/` mappát a Drive-odon, oda töltsd fel a parquet
fájlokat, és a notebookban kapcsold be a Drive-cellát.

---

## 8. lépés — Az első notebook futtatása

```bash
# A projektmappában, aktivált .venv-vel:
code .
```

VS Code-ban: `notebooks/01_data_loading_eda.ipynb` megnyitása → jobb felül
**Select Kernel** → **Python Environments** → a `.venv`-es opció.

Futtasd a cellákat sorban (`Shift+Enter`).

---

## Gyakori hibák

| Hiba | Ok és megoldás |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | A notebookot nem a `notebooks/` mappából futtatod, vagy kimaradt a `sys.path.insert(0, "..")` cella. Futtasd az első cellát. |
| `python: command not found` (Windows) | Kimaradt az „Add to PATH" pipa. Telepítsd újra a Pythont. |
| A `(.venv)` nem jelenik meg | Nem aktiváltad. Minden új terminálban újra kell. |
| VS Code nem találja a csomagokat | Rossz kernel van kiválasztva. Jobb felül Select Kernel → a `.venv`. |
| `Kaggle: 401 Unauthorized` | Rossz helyen a `kaggle.json`. Vagy töltsd le kézzel a CSV-t. |
| Kifogy a memória az adatbetöltésnél | Csökkentsd a `CFG.dev_sample_size` értékét a `config.py`-ban. |
| `CUDA out of memory` (Colab) | `BATCH_SIZE = 16` vagy `8`. |

---

## Napi rutin, ha visszaülsz dolgozni

```bash
cd ~/szakdolgozat/hotel-absa
source .venv/bin/activate        # Windowson: .venv\Scripts\Activate.ps1
code .
```

Munka végén:

```bash
git add .
git commit -m "EDA abrak elkeszultek, kuszob-erzekenysegi vizsgalat"
```

> **Commitolj minden nap**, rövid, leíró üzenettel. A védés előtt a
> `git log` egy kész idővonal lesz arról, hogyan haladt a munkád.
