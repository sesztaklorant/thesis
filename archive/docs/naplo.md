# Kutatási napló

> **Mire való ez a fájl?** Minden kódolási fázis után írj ide néhány bekezdést:
> mit csináltál, mit láttál, és **miért döntöttél úgy, ahogy**. Nem kell szépen
> fogalmazni — a lényeg, hogy rögzítsd, amíg friss.
>
> A dolgozat 6., 7., 8. és 9. fejezete lényegében ennek a naplónak a
> megtisztított változata lesz. Sokkal könnyebb most leírni, mint három hónap
> múlva visszaemlékezni rá, hogy miért éppen 6-ot választottál küszöbnek.

---

## Sablon minden fázishoz

```
### [dátum] — [fázis neve]

**Mit csináltam:**

**Milyen döntéseket hoztam, és miért:**

**Mit találtam (számokkal):**

**Mi lepett meg:**

**Mi maradt nyitva / mit kell megkérdezni a témavezetőtől:**
```

---

### 2026-__-__ — 0. fázis: környezet

**Mit csináltam:**
-

**Mi akadt el:**
-

---

### 2026-__-__ — 1. fázis: adat és EDA

**Melyik adathalmazt választottam és miért:**
-

**Licenc és hivatkozás:**
-

**Sentiment-küszöbök (neg ≤ __ / pos ≥ __), és miért ezek:**
-

**Mit mutatott a küszöb-érzékenységi vizsgálat:**
-

**Szűrési napló (hány sort vesztettem melyik lépésben):**
| Lépés | Sorok | Változás |
|---|---|---|
| kiindulás | | |
| rövid szövegek | | |
| duplikátumok | | |
| nem angol | | |

**Osztályeloszlás:**
- negative: __ % | neutral: __ % | positive: __ %
- kiegyensúlyozatlansági arány: __ x

**Mit láttam az EDA-ban:**
-

**Hol bukik el a kulcsszavas aspektusdetektálás (konkrét példák):**
-

---

### 2026-__-__ — 2. fázis: klasszikus baseline

**Eredmények (validáció):**
| Modell | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Multinomial NB | | | |
| Logistic Regression | | | |
| Linear SVM | | | |

**Legjobb modell a teszthalmazon:**
-

**Ablációs vizsgálat (RQ2) — mi számított és mi nem:**
-

**Rontott-e a tagadószavak eldobása? Mennyivel:**
-

**Hibaelemzés — mit olvastam a félreosztályozott reviewkban:**
-

---

### 2026-__-__ — 3. fázis: BERT

**Beállítások:** modell __ | max_len __ | batch __ | lr __ | epoch __

**Tanítási idő:** __ perc (__ GPU-n)

**Teszt eredmények:**
- accuracy: __ | macro-F1: __ | weighted-F1: __

**Mennyit hozott a klasszikus baseline-hoz képest:** __ pont macro-F1

**Volt-e túltanulás? Mit mutatott a tanulási görbe:**
-

**Trade-off értékelés (pontosság vs. számításigény):**
-

---

### 2026-__-__ — 4. fázis: ABSA

**HRAST tényleges oszlopszerkezete:**
-

**Annotációs séma (hogyan vannak kódolva az aspektusok és sentimentek):**
-

**Hogyan illesztettem a lexikonomat a HRAST aspektusaihoz:**
-

**Kulcsszavas baseline vs. transformer ABSA:**
| Aspektus | Baseline F1 | Transformer F1 |
|---|---|---|

**Mely aspektusok a leginkább panaszhajtók (RQ3):**
-

---

### 2026-__-__ — 5. fázis: cross-dataset

**Teljesítmény-romlás:**
| Kiértékelési adathalmaz | Macro-F1 | Romlás |
|---|---|---|
| Booking (in-domain teszt) | | — |
| 515K (ugyanaz a platform) | | |
| TripAdvisor (más platform) | | |

**Mivel magyarázom a romlást:**
-

---

## Kérdések a témavezetőnek

- [ ] Elfogadható-e, ha a külső validációhoz TripAdvisor adatot is használok
      az 515K mellett? (indoklás: az 515K is Booking.com forrású, ezért
      gyengébb általánosítási teszt)
- [ ]
- [ ]

## Megválaszolt kérdések

-
