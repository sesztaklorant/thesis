# Környezet — beállítás és dokumentáció

Ez a fájl két dolgot csinál egyszerre:

1. **Leírja a beállítás lépéseit** — ha új gépre kell átraknod a projektet, vagy ha a
   környezet valamiért tönkremegy, ez a helyreállítási útmutató.
2. **Dokumentálja a tényleges kísérleti környezetet** — a szakdolgozat módszertani
   fejezetébe és a függelékbe ebből tudsz hivatkozni. A reprodukálhatósághoz
   hozzátartozik, hogy leírod, min futottak a mérések.

---

## Az aktuális környezet (2026. szeptember)

| Komponens | Verzió / érték | Állapot |
|---|---|---|
| Operációs rendszer | Windows 11 | — |
| Python | 3.12.3 | ✅ telepítve |
| Virtuális környezet | `.venv` a projektmappában | ✅ létrehozva |
| Projekt útvonala | `C:\Users\seszt\Documents\BCE\7_felev\Szakdolgozat\THESIS\CODE` | ✅ |
| Csomagok | `requirements.txt` szerint | ✅ telepítve |
| NLTK erőforrások | stopwords, wordnet, omw-1.4, punkt | ✅ letöltve |
| Füstteszt | `tests/test_pipeline.py` | ✅ zöld |
| Git (helyi) | repó inicializálva, első commit kész | ✅ |
| GitHub | `sesztaklorant/thesis` (privát) | ⏳ push VS Code-ból |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, **4 GB VRAM** | ✅ |
| NVIDIA driver | **616.92** (Studio Driver) | ✅ frissítve |
| Driver CUDA-támogatás | **13.4** | ✅ |
| PyTorch | CUDA-s build, **cu132** csatorna | ✅ telepítve |
| transformers / accelerate / evaluate | legfrissebb | ✅ telepítve |

> **A dolgozatba:** „A kísérleteket Windows 11 rendszeren, Python 3.12 környezetben
> futtattam. A transformer-modellek finomhangolása NVIDIA GeForce RTX 3050 Ti
> Laptop GPU-n (4 GB VRAM) történt, CUDA-gyorsítással."

---

## Hol fut mi?

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  A LAPTOP (elsődleges)       │     │  GOOGLE COLAB (tartalék)     │
│                              │     │                              │
│  • adatletöltés, tisztítás   │     │  • nagy mintás végleges      │
│  • feltáró elemzés (EDA)     │ ──▶ │    futtatás                  │
│  • TF-IDF + NB/LogReg/SVM    │     │  • nagyobb modellek          │
│  • BERT finomhangolás        │ ◀── │    (RoBERTa-large stb.)      │
│    (RTX 3050 Ti, 4 GB)       │     │                              │
│  • ábrák, táblázatok         │     │  T4 GPU, 16 GB VRAM          │
│  • a dolgozat írása          │     │                              │
└──────────────────────────────┘     └──────────────────────────────┘
```

A saját GPU a **fejlesztéshez** jobb: nincs időkorlát, nincs napi kvóta, nem vesznek
el a fájlok, és nem kell parquetet fel-le tologatni. A Colab akkor jön elő, ha a
4 GB VRAM kevés — nagyobb modellnél vagy sokkal nagyobb mintánál.

---

## 1. lépés — Python

**Telepítve: 3.12.3.** Ez rendben van, nem kell más verzió — a PyTorch, a
transformers és a scikit-learn mind támogatja.

> Ne telepíts párhuzamosan másik Python-verziót. Windowson a többszörös
> Python-telepítés több bajt okoz, mint amennyi hasznot hoz: összekeverednek a
> `python` és `pip` parancsok, és nehéz kideríteni, melyik környezetbe ment a csomag.

Ellenőrzés:

```powershell
python --version        # Python 3.12.3
```

Ha új gépre telepíted: <https://www.python.org/downloads/> → a telepítőben
**pipáld be** legalul az **„Add python.exe to PATH"** opciót. Ez a leggyakoribb
hibaforrás Windowson.

---

## 2. lépés — VS Code

1. <https://code.visualstudio.com/> → telepítés
2. Extensions (bal oldali négy kocka) → telepítsd:
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)

A notebookokat közvetlenül VS Code-ban futtatod, nem böngészőben.

---

## 3. lépés — Projektmappa és virtuális környezet

### Miért kell a venv?

A `.venv` egy mappa a projekten belül, saját Python-másolattal és saját
csomagkönyvtárral. Három konkrét haszna van itt:

1. **A PyTorch nagy és ragaszkodó.** A CUDA-s telepítés ~3 GB, és rögzített
   verziójú függőségeket hoz (numpy, sympy, networkx). Ha ez a rendszer-Pythonba
   kerül, minden más projekted is ezeket kapja.
2. **Ha elromlik, 5 perc a javítás.** Törlöd a `.venv` mappát, és
   `pip install -r requirements.txt`-vel újraépíted. Rendszerszintű telepítésnél
   ugyanez a *Pythonod* megjavítását jelentené.
3. **A `requirements.txt` csak így mond igazat** — pontosan az van benne, ami a
   szakdolgozathoz kell, nem az összes valaha telepített csomagod.

Ára: minden új terminálban aktiválni kell. VS Code-ban ez nem érint, mert ott a
kernelválasztás egyszeri.

### A mappa helye

Ékezet és szóköz nélküli útvonalon legyen — a Python egyes könyvtárai megbotlanak
az ékezetes elérési utakon. A jelenlegi útvonal ezt teljesíti.

### Létrehozás és aktiválás

```powershell
cd C:\Users\seszt\Documents\BCE\7_felev\Szakdolgozat\THESIS\CODE
python -m venv .venv
.venv\Scripts\Activate.ps1
```

A terminál elején meg kell jelennie a `(.venv)` előtagnak.

> **Ha „a szkriptek futtatása le van tiltva" hibát kapsz**, futtasd egyszer:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> **Fontos:** a `.venv` a projektmappával **egy szinten** legyen, vagyis a
> `requirements.txt` mellett. Ha egy szinttel feljebb hozod létre, működik ugyan,
> de a `.gitignore` nem kezeli, és a parancsok is kényelmetlenebbek lesznek.

### Csomagok telepítése

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Figyelem a pip frissítésére:** `python -m pip` kell, **nem** `pip install --upgrade pip`.
> Windowson a `pip.exe` zárolva van futás közben, ezért nem tudja lecserélni saját
> magát, és hibaüzenettel visszautasítja. A `python -m pip` alak megkerüli ezt,
> mert a Python futtatja a pip modult.

### NLTK nyelvi erőforrások

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

Ezek nélkül a kód tartalék megoldásra vált (beépített minimális stopword-lista és
egyszerűsített lemmatizáló), és figyelmeztetést ír ki. A `preprocessing.py`
szándékosan így van megírva, de élesben a rendes NLTK-erőforrásokat használd.

### Ellenőrzés — füstteszt

```powershell
python tests\test_pipeline.py
```

A végén ezt kell látnod:

```
==================================================================
 MINDEN TESZT SIKERES — a környezeted készen áll.
==================================================================
```

A teszt szintetikus adaton futtatja végig a teljes pipeline-t: tisztítás →
címkézés → 60/20/20 felosztás → három modell tanítása → kiértékelés → ABSA.

> **A teszt 1,000-es pontossága nem jelent semmit** a valódi teljesítményre nézve.
> A szintetikus adat hat fix mondatsablonból áll, tehát triviálisan szétválasztható.
> A teszt csak azt bizonyítja, hogy a *kód* hibátlanul végigfut. A valódi Booking-adaton
> 0,70–0,80 körüli macro-F1 a reális tartomány.

> Ha a futtatás nem ír ki NLTK-figyelmeztetést, az jó jel: az erőforrások a helyükön
> vannak, és a valódi WordNet lemmatizáló fut.

---

## 4. lépés — Git és GitHub

### Helyi verziókövetés (GitHub-fiók nélkül is működik)

A Git és a GitHub két külön dolog: a Git a verziókövető a gépeden, a GitHub csak
tárhely. A helyi commitok az első pillanattól működnek.

```powershell
git config --global user.name "Sesztak Lorant"
git config --global user.email "sesztaklorant@gmail.com"

git init
git add .
git commit -m "Kezdeti projektvaz: config, src modulok, notebookok, fustteszt zold"
```

A `.gitignore` gondoskodik róla, hogy az adatok (`data/`) és a betanított modellek
(`results/models/`) **ne** kerüljenek a repóba — méret és licenc miatt egyaránt.

### GitHub-repó összekötése

```powershell
git remote add origin https://github.com/sesztaklorant/thesis.git
git branch -M main
git push -u origin main
```

> **A jelszavas belépés nem működik.** A GitHub 2021 óta nem fogad el jelszót
> Git-műveletekhez. Ha a terminál jelszót kér és beírod a GitHub-jelszavadat,
> ezt kapod:
> ```
> remote: Invalid username or token. Password authentication is not supported.
> fatal: Authentication failed
> ```

**A legegyszerűbb megoldás — VS Code:**

1. `code .` a projektmappában
2. Bal oldalt legalul a **fiók ikon** (emberke) → **Sign in with GitHub**
3. Böngészőben engedélyezed, visszadob VS Code-ba
4. **Source Control** panel (`Ctrl+Shift+G`) → **Sync Changes** / push gomb

**Ha parancssorból akarod**, két út van:

*a) Böngészős hitelesítés*

```powershell
git config --global credential.helper manager
git push -u origin main
```

Felugrik egy böngészőablak a GitHub-belépéshez.

*b) Személyes hozzáférési token (PAT)*

GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)**
→ Generate new token → jogosultság: **`repo`** → másold ki azonnal (többé nem látod).

Push-nál: felhasználónév = `sesztaklorant`, jelszó helyett **illeszd be a tokent**.
A PowerShell nem mutatja a beillesztést — ez normális, nyomj Entert.

> Ha korábban elmentődött a rossz jelszó: Windows **Hitelesítőadat-kezelő** →
> **Windows hitelesítő adatok** → töröld a `git:https://github.com` bejegyzést.

---

## 5. lépés — GPU és PyTorch

### 5.1 A GPU ellenőrzése

```powershell
nvidia-smi
```

A kimenetből három dolog számít:

- **Memory-Usage jobb oldali száma**: `4096MiB` = 4 GB VRAM. Ez a szűk keresztmetszet.
- **CUDA Version** (jobb felül): a driver által támogatott **legmagasabb** CUDA-verzió.
  Nem azt jelenti, hogy az van telepítve.
- **Processes** lista: mi eszi épp a GPU-t.

### 5.2 Driver

Telepítve: **NVIDIA Studio Driver 616.92**, CUDA-támogatás **13.4**.

**Miért Studio és nem Game Ready?** CUDA szempontból a kettő egyenértékű, a
PyTorch mindkettővel ugyanúgy fut. A Studio hosszabb tesztelési cikluson megy át
és a stabilitást helyezi előre, a Game Ready pedig a megjelenésnapi
játéktámogatást. Számítási munkához a Studio a kényelmesebb.

Frissítés: <https://www.nvidia.com/download/index.aspx> → a telepítőben **Egyéni
(Custom)** → pipáld be a **„Tiszta telepítés végrehajtása"** opciót. Ez kitakarítja
a régi driver maradékait. Figyelem: a tiszta telepítés törli az NVIDIA vezérlőpulton
beállított egyedi profilokat.

### 5.3 PyTorch telepítése

**Ne telepíts CUDA Toolkitet az NVIDIA oldaláról** — erre nincs szükség, és sokan
feleslegesen végigcsinálják. A pip-es PyTorch-csomagok hozzák magukkal a szükséges
CUDA-futtatókörnyezetet; a gépre csak a videokártya-driver kell.

**Nem kell a driveréddel pontosan egyező CUDA-build sem.** A driverek visszafelé
kompatibilisek: egy 13.4-et támogató driver gond nélkül futtat cu12.x-re vagy
cu13.2-re épült buildeket. A stabil PyTorch-kiadások jellemzően egy-két verzióval
a legfrissebb CUDA alatt járnak — ez normális.

A telepített parancs (a <https://pytorch.org/get-started/locally/> selectorából,
Stable / Windows / Pip / Python / CUDA 13.2):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

> A `cu132` rész verzióváltáskor változik — új gépen mindig a selectorból vedd a
> parancsot, ne ebből a fájlból másold. A `torchaudio` nem kell; a `torchvision`
> szövegfeldolgozáshoz szintén nem, de ártalmatlan.
>
> **Az aktivált venv-ben futtasd** (legyen ott a `(.venv)` előtag). ~3 GB letöltés.

### 5.4 Ellenőrzés

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Helyes kimenet:

```
2.x.x+cu132        <- a lényeg a +cu rész
True
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

Ha `False` jön, vagy a verziószámban nincs `+cu`, akkor CPU-s build települt
(jellemzően azért, mert lemaradt az `--index-url`):

```powershell
pip uninstall torch torchvision
```

és telepítsd újra az index-url-lel.

### 5.5 Transformer csomagok

```powershell
pip install transformers accelerate evaluate
```

### 5.6 A 4 GB VRAM kezelése — FONTOS

A `03_bert_colab.ipynb` alapértelmezései Colab T4-re (16 GB) vannak hangolva.
**Helyben ezek elszállnak** `CUDA out of memory` hibával. A beállítási cellában
írd át:

```python
BATCH_SIZE = 8          # 32 helyett
GRAD_ACCUM = 4          # 8 × 4 = 32 effektív batch
```

és a `TrainingArguments`-ben:

```python
per_device_train_batch_size=BATCH_SIZE,
gradient_accumulation_steps=GRAD_ACCUM,
per_device_eval_batch_size=16,
fp16=True,
```

**Miért pont így?** A `gradient_accumulation_steps` négy kis batch gradiensét
gyűjti össze, mielőtt lép egyet az optimalizáló. Az eredmény matematikailag
**ugyanaz, mint egy 32-es batch**, csak több lépésben számolja. Ez azért fontos,
mert így a helyi és a Colabos futtatásod **összehasonlítható marad** — ha a batch
méretét változtatnád, a tanulási ráta is máshogy viselkedne, és nem lehetne
megmondani, a modell vagy a hiperparaméter okozta-e a különbséget.

Ha 8-cal is elszáll: `BATCH_SIZE = 4`, `GRAD_ACCUM = 8`.

**Tanítás előtt zárd be a VRAM-fogyasztókat.** 4 GB-nál minden megabájt számít:

- **Epic Games Launcher** — nem elég az ablakot bezárni, a tálca értesítési
  területén tovább fut; ott kell kilépni belőle
- **Chrome** — simán elvisz fél gigát
- **NVIDIA App overlay**, ha telepítve van

Ellenőrzés közvetlenül tanítás előtt:

```powershell
nvidia-smi
```

A Memory-Usage bal oldali száma legyen 0 MiB közeli.

---

## 6. lépés — Kaggle API

A Booking.com adathalmaz Hugging Face-ről jön, ahhoz nem kell fiók. A TripAdvisor,
az 515K és a HRAST Kaggle-ről.

1. Regisztrálj: <https://www.kaggle.com>
2. Profilkép → **Settings** → **API** → **Create New Token**
3. A letöltött `kaggle.json` helye: `C:\Users\seszt\.kaggle\kaggle.json`

> **Ha ez bonyolultnak tűnik:** teljesen rendben van kézzel letölteni a CSV-ket a
> böngészőből, és bemásolni a `data/raw/` mappába. A `data_loading.py` mindkét
> utat megtalálja — először a helyi fájlt keresi, és csak utána próbál letölteni.

---

## 7. lépés — Google Colab (tartalék)

Saját GPU-val a Colab már nem kötelező, de két esetben jól jön: ha a 4 GB kevés
(nagyobb modell), vagy ha nagyon nagy mintán futtatnál véglegeset.

1. <https://colab.research.google.com> → **File → Upload notebook** →
   `notebooks/03_bert_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU** → Save
3. Töltsd fel a `data/processed/` parquet fájljait (bal oldali mappa ikon),
   vagy csatold a Google Drive-ot

**Colab korlátok:** ~90 perc inaktivitás után lecsatlakozik (és a fájlok elvesznek),
napi GPU-kvóta kb. 3–4 óra. Mindig töltsd le, amit előállítasz.

Ha Colabon futtatod, a helyi beállításokat **állítsd vissza** `BATCH_SIZE = 32`,
`GRAD_ACCUM = 1` értékre — ott elfér.

---

## 8. lépés — Az első notebook

```powershell
code .
```

VS Code-ban: `notebooks/01_data_loading_eda.ipynb` megnyitása → jobb felül
**Select Kernel** → **Python Environments** → a `.venv`-es opció.

Futtasd a cellákat sorban (`Shift+Enter`).

---

## Hibaelhárítás

| Hiba | Ok és megoldás |
|---|---|
| `ERROR: To modify pip, please run...` | `pip install --upgrade pip` helyett `python -m pip install --upgrade pip` |
| `ModuleNotFoundError: No module named 'src'` | Nem a projektgyökérből futtatod, vagy kimaradt a notebook első cellája (`sys.path.insert`) |
| A `(.venv)` nem jelenik meg | Nem aktiváltad. Minden új terminálban újra kell. |
| „a szkriptek futtatása le van tiltva" | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| VS Code nem találja a csomagokat | Rossz kernel. Jobb felül Select Kernel → a `.venv`. |
| `Password authentication is not supported` | GitHub-jelszó nem jó. VS Code-os belépés vagy PAT — lásd 4. lépés. |
| `torch.cuda.is_available()` = `False` | CPU-s build települt. `pip uninstall torch torchvision`, majd újra `--index-url`-lel. |
| `CUDA out of memory` | `BATCH_SIZE = 4`, `GRAD_ACCUM = 8`; zárd be az Epic Launchert és a Chrome-ot |
| A tanítás órákig tart | Nem a GPU-n fut. Ellenőrizd a `torch.cuda.is_available()`-t. |
| NLTK-figyelmeztetés a füsttesztben | Futtasd az `nltk.download(...)` parancsot a 3. lépésből. |
| `Kaggle: 401 Unauthorized` | Rossz helyen a `kaggle.json`, vagy töltsd le kézzel a CSV-t a `data/raw/`-ba. |
| Kifogy a memória adatbetöltésnél | Csökkentsd a `CFG.dev_sample_size` értékét a `src/config.py`-ban. |

---

## Napi rutin

Munka elején:

```powershell
cd C:\Users\seszt\Documents\BCE\7_felev\Szakdolgozat\THESIS\CODE
.venv\Scripts\Activate.ps1
code .
```

Munka végén:

```powershell
git add .
git commit -m "EDA abrak elkeszultek, kuszob-erzekenysegi vizsgalat"
git push
```

> **Commitolj minden munkanap végén**, rövid, leíró üzenettel. A védés előtt a
> `git log` kész idővonal lesz arról, hogyan haladt a munkád — és a témavezetőnek
> is meg tudod mutatni.

---

## Környezet újraépítése nulláról

Ha új gépre viszed, vagy valami menthetetlenül elromlik:

```powershell
# 1. Projekt letöltése
git clone https://github.com/sesztaklorant/thesis.git
cd thesis

# 2. Virtuális környezet
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Csomagok
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"

# 4. GPU (opcionális) — a parancsot a pytorch.org selectorából vedd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install transformers accelerate evaluate

# 5. Ellenőrzés
python tests\test_pipeline.py
```

Az adatok nincsenek a repóban — azokat a notebookok újra letöltik, vagy kézzel
teszed a `data/raw/` mappába.
