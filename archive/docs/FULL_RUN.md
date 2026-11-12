# Teljes futtatás (≈203 ezer vélemény) — lépésről lépésre

## Mit kell újratanítani?

| Modell | Újratanítás? | Miért |
|---|---|---|
| (A) zero-shot modellek | nincs tanítás | előtanított modellek, csak címkéznek |
| (B) BERT a HRAST-on | **NEM** | csak a HRAST-ból tanul, a Booking-adattól független; a notebook `RETRAIN_B = False` mellett újrahasznosítja |
| (C) önálló tanítás (self-training) | **NEM** (alapból) | a pszeudo-címkék száma osztályonként korlátos, a teljes adaton sem lenne nagyobb a tanítóhalmaz; csak összehasonlító megközelítés; `RETRAIN_C = False` |
| 2/a aspektusmodell (HRAST) | **NEM** | csak a HRAST-ból tanul; a notebook automatikusan kihagyja, ha létezik |
| TF-IDF „tanuló” modellek (04) | IGEN, automatikusan | a végső címkékből tanulnak, ezek a teljes adaton újak; néhány perc |
| 2/b altémák (05) | IGEN, automatikusan | az új egységeken fürtöz; gyors |

Következmény: a 03-as notebook gold-eredményei (A, B, C, végső címke) **gyakorlatilag változatlanok** maradnak.

## 0. Előkészítés

1. Laptop töltőn; Windows: *Beállítások → Rendszer → Energiaellátás → Alvó állapot: Soha* (töltőn).
2. Zárj be minden notebookot a VS Code-ban; más GPU-t használó programot is.
3. Szabad hely: legalább kb. 10 GB.

## 1. Befagyasztás (git)

A VS Code termináljában, a CODE mappában:

```
git status
git add -A
git commit -m "Layer 2 final (v5), 20k sample"
git tag v0.3-sample20k
git push
git push origin v0.3-sample20k
```

## 2. Archiválás (a CODE mappán kívülre)

```
New-Item -ItemType Directory ..\archive_sample20k
Copy-Item -Recurse results ..\archive_sample20k\results
Copy-Item -Recurse data\labels ..\archive_sample20k\labels
Copy-Item -Recurse data\processed ..\archive_sample20k\processed
Copy-Item -Recurse data\annotation ..\archive_sample20k\annotation
```

Ellenőrzés: `Get-ChildItem ..\archive_sample20k` — négy mappa.

## 3. Config (`src/config.py`) — CSAK ez a két sor változik

```python
labelling_sample_size: int | None = None      # volt: 20_000
topic_version: str = "full"                   # volt: "v5"
```

Minden más marad (neutral band 0,25, final = B stb.). Mentés.

## 4. Notebookok, sorrendben — mindegyiknél: Restart → Run All

| # | Notebook | Mire figyelj | Becsült idő |
|---|---|---|---|
| 01 | data_preparation_eda | a végén: `working set: ~203 000 rows` (= prepared) | 5–10 perc |
| 02 | bert_labelling | `RETRAIN_B = False`, `RETRAIN_C = False`; kiírja: „(B) is already trained … reused”, „(C) already exists — reused” | 2–4 óra |
| 03 | label_validation | a gold-táblázat számai ugyanazok, mint a 20k-s futásban (legfeljebb a 3. tizedesjegyben térhetnek el — fp16 kerekítés) | 5 perc |
| 04 | text_analysis_classical | TF-IDF modellek tanulnak (gold kizárva) | 10–20 perc |
| 05 | thematic_layer, `SENTIMENT = "negative"` | egységek: kb. 10× több; a B és az aspektus-besorolás darabolva fut | 1–2 óra |
| 05 | thematic_layer, `SENTIMENT = "positive"` | a `SENTIMENT` cellában átírod, majd Restart → Run All (az 1–3. rész a cache-ből gyors) | 30–60 perc |

Ha bármi megszakad (lefagy, újraindul a gép): ugyanazt a notebookot **Restart → Run All** — a cache-elt
darabokat nem számolja újra, ott folytatja, ahol abbamaradt.

A 05-ös kézi ellenőrző Excelt (`topic_check_*_full.xlsx`) MÉG NE töltsd ki.

## 5. Ellenőrzőlista a végén

- [ ] 03: gold — B: accuracy ≈ 0,8045, macro-F1 ≈ 0,6405 (band 0,25); ha 0,01-nél jobban eltér, szólj
- [ ] 05: `05_negative_full_*` és `05_positive_full_*` táblák és ábrák léteznek
- [ ] commit: `git add -A` → `git commit -m "Full run"` (tag csak a kézi ellenőrzés után: `v1.0-full`)

## 6. Utána (full2) — elnevezés és lezárás

1. Kézi ellenőrzés: kész (pontosság: pozitív 0,927, negatív 0,840).
2. Nevek: `src/topic_names.py` (188 + 191 fürt → 167 + 167 elnevezett altéma). A notebook 05 a `full2` verziónál
   automatikusan alkalmazza; ha a fürtök megváltoztak, az ujjlenyomat-ellenőrzés hibát dob.
3. Notebook 05 újrafuttatása: `SENTIMENT = "negative"` → Restart → Run All, majd `"positive"` → Restart → Run All
   (minden cache-ből jön; a `topic_check_*_full2.xlsx` létezik, ezért csak kiértékel, nem írja felül).
4. `git add -A` → `git commit -m "Topic names, v1.0"` → `git tag v1.0-full` → `git push` → `git push origin v1.0-full`.
