"""
bert_labelling.py — Szövegalapú szentiment-címkék előállítása BERT-modellekkel.
                    Producing text-based sentiment labels with BERT models.

HU: A PROBLÉMA: a Booking-adatban nincs szentiment-címke, csak 1–10-es
    vendégpontszám, és a pontszám egyszerű vágása NEM érvényes címke (a pontszám
    a teljes tartózkodás értékelése, a szöveg hangulata ettől eltérhet).
    A MEGOLDÁS: a címkét a SZÖVEGBŐL állítjuk elő, három független úton:

      (A) Zero-shot   — kész, mások által már finomhangolt szentiment-modellek;
                        mi nem tanítunk semmit.
      (B) Transzfer   — BERT finomhangolása a kézzel annotált HRAST-on, majd
                        alkalmazása a Booking-mondatokra.
      (C) Önképzés    — BERT finomhangolása magán a Booking-adaton, az (A) és (B)
                        legbiztosabb címkéit tanítójelnek használva (pseudo-label).

    Minden megközelítés UGYANAZT a kimenetet adja (megközelítés-előtaggal):
        <X>_p_negative, <X>_p_neutral, <X>_p_positive   — osztályvalószínűségek
        <X>_label        — negative / neutral / positive
        <X>_confidence   — a legnagyobb valószínűség
        <X>_score_cont   — folytonos 1–10 pontszám
        <X>_score10      — egész 1–10 pontszám
    Így a három eredmény közvetlenül összevethető (label_validation.py), és a
    végső címke többségi szavazással képezhető (combine_approaches).

EN: THE PROBLEM: the Booking data has no sentiment label, only a 1–10 guest
    score, and a simple cut of the score is NOT a valid label (the score rates
    the whole stay, the tone of the text may differ).
    THE SOLUTION: the label is produced from the TEXT, in three independent ways:

      (A) Zero-shot   — ready sentiment models fine-tuned by others; we train nothing.
      (B) Transfer    — BERT fine-tuned on human-annotated HRAST, then applied
                        to the Booking sentences.
      (C) Self-train  — BERT fine-tuned on the Booking data itself, using the most
                        confident labels of (A) and (B) as the training signal.

    Every approach returns the SAME output (with an approach prefix, see above),
    so the three results are directly comparable (label_validation.py) and the
    final label can be formed by majority vote (combine_approaches).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import CFG, PATHS

log = logging.getLogger(__name__)

LABELS: list[str] = list(CFG.labels)
PROB_SUFFIXES: list[str] = [f"p_{label}" for label in LABELS]
STARS: np.ndarray = np.arange(1, 6, dtype=float)


# ==========================================================================
# 1) TISZTA FÜGGVÉNYEK (torch nélkül tesztelhetők)
#    PURE FUNCTIONS (testable without torch)
# ==========================================================================
def round_half_up(values) -> np.ndarray:
    """
    HU: Kerekítés .5-ről felfelé. A numpy np.round "banki" kerekítést használ
        (4.5 -> 4, 5.5 -> 6), ami a pontszámnál következetlen lenne.
    EN: Round half up. numpy's np.round uses "banker's" rounding
        (4.5 -> 4, 5.5 -> 6), which would be inconsistent for the score.
    """
    return np.floor(np.asarray(values, dtype=float) + 0.5)


def polarity_from_probs(probs: np.ndarray) -> np.ndarray:
    """
    HU: Polaritás-index = P(pozitív) − P(negatív), értéke [−1, +1].
    EN: Polarity index = P(positive) − P(negative), in [−1, +1].
    """
    probs = np.asarray(probs, dtype=float)
    return probs[:, LABELS.index("positive")] - probs[:, LABELS.index("negative")]


def polarity_to_score(polarity) -> tuple[np.ndarray, np.ndarray]:
    """
    HU: Polaritás [−1, +1] -> 1–10 skála, lineárisan:  score = 1 + 4.5 · (polaritás + 1)
          −1 (biztosan negatív) -> 1,  0 (semleges / kiegyenlített) -> 5.5,  +1 -> 10
    EN: Polarity [−1, +1] -> 1–10 scale, linearly:  score = 1 + 4.5 · (polarity + 1)
          −1 (surely negative) -> 1,  0 (neutral / balanced) -> 5.5,  +1 -> 10

    Returns: (continuous score, integer score)
    """
    cont = 1.0 + 4.5 * (np.clip(np.asarray(polarity, dtype=float), -1, 1) + 1.0)
    return cont, np.clip(round_half_up(cont), 1, 10).astype(int)


def stars_to_score(star_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HU: 1–5 csillagos modell valószínűségei -> 1–10 pontszám.
          1) várható csillagszám:  E = Σ k · P(k csillag)       ∈ [1, 5]
          2) lineáris átskálázás:  score = 1 + 9 · (E − 1) / 4   ∈ [1, 10]
          3) egészre kerekítés (.5 felfelé)
        A várható érték (és nem a legvalószínűbb csillag) azért jobb, mert a
        modell bizonytalanságát is megőrzi: 50% 4★ + 50% 5★ -> 4,5★ -> 8,9.
    EN: Probabilities of a 1–5 star model -> 1–10 score.
          1) expected stars:      E = Σ k · P(k stars)          ∈ [1, 5]
          2) linear rescaling:    score = 1 + 9 · (E − 1) / 4    ∈ [1, 10]
          3) round to integer (half up)
        The expected value (not the most likely star) keeps the model's
        uncertainty: 50% 4★ + 50% 5★ -> 4.5★ -> 8.9.

    Returns: (expected stars, continuous score, integer score)
    """
    star_probs = np.asarray(star_probs, dtype=float)
    expected = star_probs @ STARS
    cont = 1.0 + 9.0 * (expected - 1.0) / 4.0
    return expected, cont, np.clip(round_half_up(cont), 1, 10).astype(int)


def stars_to_3class(expected_stars: np.ndarray) -> np.ndarray:
    """
    HU: CSAK DIAGNOSZTIKA: várható csillag -> 3 osztály (≤2,5 neg, ≥3,5 poz, köztes semleges).
        Az (A) két modelljének belső egyezését méri; nem ez a fő címke.
    EN: DIAGNOSTIC ONLY: expected stars -> 3 classes (≤2.5 neg, ≥3.5 pos, else neutral).
        Measures the internal agreement of the two (A) models; not the main label.
    """
    expected_stars = np.asarray(expected_stars, dtype=float)
    return np.where(expected_stars <= 2.5, "negative",
                    np.where(expected_stars >= 3.5, "positive", "neutral"))


def probs_to_frame(probs: np.ndarray, index, prefix: str) -> pd.DataFrame:
    """
    HU: (n, 3) valószínűségmátrix -> egységes kimeneti tábla az adott előtaggal.
    EN: (n, 3) probability matrix -> uniform output table with the given prefix.
    """
    probs = np.asarray(probs, dtype=float)
    out = pd.DataFrame(probs, index=index, columns=[f"{prefix}_{s}" for s in PROB_SUFFIXES])
    out[f"{prefix}_label"] = np.array(LABELS)[probs.argmax(axis=1)]
    out[f"{prefix}_confidence"] = probs.max(axis=1)
    cont, integer = polarity_to_score(polarity_from_probs(probs))
    out[f"{prefix}_score_cont"] = cont
    out[f"{prefix}_score10"] = integer
    return out


def aggregate_sentences(sentence_probs: np.ndarray, review_ids: pd.Series,
                        mixed_threshold: float = 0.6) -> tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    HU: Mondatszintű valószínűségek -> véleményszint (a mondatok átlaga).
        Mellékterméke az "is_mixed" jelző: van a véleményben legalább egy biztosan
        pozitív ÉS egy biztosan negatív mondat. A Booking-szövegeknél ez gyakori
        (dicséret + panasz), és pont ezért nem lehet a pontszámot egyszerűen vágni.
    EN: Sentence-level probabilities -> review level (mean of the sentences).
        By-product: the "is_mixed" flag — the review has at least one confidently
        positive AND one confidently negative sentence. Common in Booking texts
        (praise + complaint), which is exactly why the score cannot simply be cut.

    Returns: (review-level probs, review_id index, extra columns: n_sentences, is_mixed)
    """
    frame = pd.DataFrame(np.asarray(sentence_probs, dtype=float), columns=LABELS)
    frame["review_id"] = np.asarray(review_ids)
    frame["strong_pos"] = frame["positive"] >= mixed_threshold
    frame["strong_neg"] = frame["negative"] >= mixed_threshold

    grouped = frame.groupby("review_id", sort=False)
    mean_probs = grouped[LABELS].mean()
    extra = pd.DataFrame({
        "n_sentences": grouped.size(),
        "is_mixed": grouped["strong_pos"].any() & grouped["strong_neg"].any(),
    })
    return mean_probs.to_numpy(), mean_probs.index, extra


def parse_model_labels(id2label: dict) -> tuple[str, list]:
    """
    HU: A modell saját címkéinek értelmezése (config.id2label).
          "negative/neutral/positive" (bármilyen kis-/nagybetűvel) -> ("3class", [kanonikus címke indexenként])
          "1 star" … "5 stars"                                     -> ("stars",  [csillagszám indexenként])
    EN: Interpret the model's own labels (config.id2label).
          "negative/neutral/positive" (any case)  -> ("3class", [canonical label per index])
          "1 star" … "5 stars"                    -> ("stars",  [number of stars per index])
    """
    ordered = [str(id2label[i]) for i in sorted(id2label, key=int)]
    lowered = [name.strip().lower() for name in ordered]
    short = {"neg": "negative", "neu": "neutral", "pos": "positive"}
    canonical = [short.get(name[:3]) if name[:3] in short else None for name in lowered]
    if len(ordered) == 3 and set(canonical) == set(LABELS):
        return "3class", canonical

    stars = [re.match(r"^\s*(\d)\s*star", name) for name in lowered]
    if len(ordered) == 5 and all(stars):
        return "stars", [int(m.group(1)) for m in stars]

    if len(ordered) == 3 and all(name.startswith("label_") for name in lowered):
        # HU: Régi modellek "LABEL_0/1/2" néven: a szokásos sorrend neg/neu/pos — ELLENŐRIZD a modellkártyán!
        # EN: Old models named "LABEL_0/1/2": the usual order is neg/neu/pos — CHECK the model card!
        log.warning("Model labels are %s; assuming negative/neutral/positive order.", ordered)
        return "3class", LABELS.copy()
    raise ValueError(f"Unsupported model labels: {ordered}")


# ==========================================================================
# 2) INFERENCIA / INFERENCE
# ==========================================================================
def pick_device(prefer: str | None = None) -> str:
    """HU: cuda > mps (Apple) > cpu. / EN: cuda > mps (Apple) > cpu."""
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformerScorer:
    """
    HU: Egy Hugging Face szekvencia-osztályozó modell köré épített, kötegelt
        előrejelző. A kimenet MINDIG a kanonikus sorrendben van:
          3 osztályos modell -> (n, 3) negative/neutral/positive
          csillagos modell   -> (n, 5) 1★ … 5★
    EN: A batched predictor around a Hugging Face sequence classifier. The output
        is ALWAYS in canonical order:
          3-class model -> (n, 3) negative/neutral/positive
          star model    -> (n, 5) 1★ … 5★

    HU: Gyorsítások: (1) a szövegeket hossz szerint rendezzük, így egy kötegen
        belül alig kell kitölteni ([PAD]); (2) GPU-n fp16 (autocast).
    EN: Speed-ups: (1) texts are sorted by length, so little padding is needed
        within a batch; (2) fp16 on GPU (autocast).
    """

    def __init__(self, model_name_or_path: str, device: str | None = None,
                 max_length: int = CFG.max_length, batch_size: int = CFG.inference_batch_size,
                 fp16: bool = True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = str(model_name_or_path)
        self.device = pick_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = fp16 and self.device == "cuda"

        log.info("Loading model: %s (device: %s, fp16: %s)", self.name, self.device, self.use_fp16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()
        self.kind, self.index_labels = parse_model_labels(self.model.config.id2label)
        self._torch = torch

        # HU: Permutáció a modell kimeneti sorrendjéből a kanonikus sorrendbe.
        # EN: Permutation from the model's output order to the canonical order.
        target = LABELS if self.kind == "3class" else list(range(1, 6))
        self._order = [self.index_labels.index(t) for t in target]
        log.info("Model type: %s | labels: %s", self.kind, self.index_labels)

    def predict_proba(self, texts, show_progress: bool = True) -> np.ndarray:
        torch = self._torch
        texts = ["" if t is None else str(t) for t in texts]
        order = np.argsort([len(t) for t in texts])
        result = np.zeros((len(texts), len(self._order)), dtype=np.float32)

        n_batches = int(np.ceil(len(texts) / self.batch_size))
        started = time.time()
        for b in range(n_batches):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(**enc).logits
                else:
                    logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            result[idx] = probs[:, self._order]
            if show_progress and (b + 1) % 50 == 0:
                rate = (b + 1) * self.batch_size / (time.time() - started)
                log.info("  %s: %d/%d batches (%.0f texts/s)", self.name.split("/")[-1],
                         b + 1, n_batches, rate)
        return result


def _texts_fingerprint(texts: pd.Series) -> str:
    """HU: Rövid ujjlenyomat a bemenetről (a cache érvényességéhez). / EN: Short fingerprint of the input (cache validity)."""
    h = hashlib.sha1()
    h.update(str(len(texts)).encode())
    for t in texts.iloc[:: max(1, len(texts) // 200)]:
        h.update(str(t).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]


def _model_tag(scorer) -> str:
    """
    HU: A modell azonosítója a cache-hez: név + (helyi mappánál) a legutóbbi módosítás ideje.
    EN: Model identifier for the cache: name + (for a local folder) the latest modification time.
    """
    name = str(getattr(scorer, "name", type(scorer).__name__ if scorer is not None else "none"))
    path = Path(name)
    if path.is_dir():
        mtimes = [f.stat().st_mtime for f in path.iterdir() if f.is_file()]
        name += f"@{max(mtimes, default=0):.0f}"
    return name


def run_chunked(texts: pd.Series, predict_fn: Callable[[list[str]], np.ndarray],
                cache_name: str | None, chunk_size: int = CFG.inference_chunk_size) -> np.ndarray:
    """
    HU: Darabolt, FOLYTATHATÓ előrejelzés. Minden darab eredménye .npy fájlba kerül
        (data/labels/cache/<név>/); megszakadt futás (pl. Colab-lekapcsolás) után
        újraindítva a kész darabokat betölti, csak a hiányzókat számolja.
        A cache ujjlenyomata a bemenetből ÉS a modellből (név + módosítási idő) áll:
        ha bármelyik változik (pl. újratanított (B) modell), a régi darabok törlődnek.
    EN: Chunked, RESUMABLE prediction. Each chunk is saved to a .npy file
        (data/labels/cache/<name>/); after an interrupted run (e.g. a Colab
        disconnect) the finished chunks are loaded, only the missing ones are
        computed. The cache fingerprint covers the input AND the model (name +
        modification time): if either changes (e.g. a retrained (B) model), the
        old chunks are deleted.
    """
    if cache_name is None:
        return predict_fn(texts.tolist())

    folder = PATHS.labels / "cache" / cache_name
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = {"n_texts": int(len(texts)), "chunk_size": int(chunk_size),
                "fingerprint": _texts_fingerprint(texts),
                "model": _model_tag(getattr(predict_fn, "__self__", None))}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        log.warning("Cache '%s' is stale (input or model changed) — recomputing.", cache_name)
        for old_chunk in folder.glob("chunk_*.npy"):
            old_chunk.unlink()
    manifest_path.write_text(json.dumps(manifest))

    parts = []
    n_chunks = int(np.ceil(len(texts) / chunk_size))
    for c in range(n_chunks):
        path = folder / f"chunk_{c:04d}.npy"
        if path.exists():
            parts.append(np.load(path))
            continue
        chunk = texts.iloc[c * chunk_size:(c + 1) * chunk_size].tolist()
        log.info("[%s] chunk %d/%d (%d texts)", cache_name, c + 1, n_chunks, len(chunk))
        probs = predict_fn(chunk)
        np.save(path, probs)
        parts.append(probs)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))


def _predict_reviews(df: pd.DataFrame, scorer, text_col: str, granularity: str,
                     cache_name: str | None) -> tuple[np.ndarray, pd.Index, pd.DataFrame | None]:
    """
    HU: Közös mag: véleményszintű VAGY mondatszintű (+ aggregált) előrejelzés.
    EN: Shared core: review-level OR sentence-level (+ aggregated) prediction.
    """
    from .preprocessing import explode_to_sentences

    ids = df["review_id"]
    if granularity == "review":
        probs = run_chunked(df[text_col].fillna(""), scorer.predict_proba, cache_name)
        return probs, pd.Index(ids), None
    if granularity == "sentence":
        sentences = explode_to_sentences(df, text_col=text_col)
        log.info("%d reviews -> %d sentences", len(df), len(sentences))
        probs = run_chunked(sentences["sentence"], scorer.predict_proba,
                            f"{cache_name}_sent" if cache_name else None)
        if probs.shape[1] != 3:
            raise ValueError("Sentence aggregation is implemented for 3-class models only.")
        review_probs, index, extra = aggregate_sentences(probs, sentences["review_id"])
        return review_probs, index, extra
    raise ValueError(f"granularity must be 'review' or 'sentence', not {granularity!r}")


# ==========================================================================
# 3) A HÁROM MEGKÖZELÍTÉS / THE THREE APPROACHES
# ==========================================================================
def label_with_classifier(df: pd.DataFrame, model_name_or_path: str, prefix: str,
                          text_col: str = "text_bert", granularity: str = "review",
                          cache_name: str | None = None, scorer=None) -> pd.DataFrame:
    """
    HU: Általános címkéző: egy 3 osztályos modellel előrejelez, és az egységes
        kimeneti táblát adja vissza (index: review_id). (B) és (C) ezt használja.
    EN: Generic labeller: predicts with a 3-class model and returns the uniform
        output table (index: review_id). Used by (B) and (C).
    """
    scorer = scorer or TransformerScorer(model_name_or_path)
    if scorer.kind != "3class":
        raise ValueError(f"{model_name_or_path} is not a 3-class model ({scorer.kind}).")
    probs, index, extra = _predict_reviews(df, scorer, text_col, granularity, cache_name)
    out = probs_to_frame(probs, index, prefix)
    if extra is not None:
        out[f"{prefix}_n_sentences"] = extra["n_sentences"].reindex(index).to_numpy()
        out[f"{prefix}_is_mixed"] = extra["is_mixed"].reindex(index).to_numpy()
    out.index.name = "review_id"
    return out


def predict_sentences(df: pd.DataFrame, model_name_or_path: str, prefix: str,
                      text_col: str = "text_bert", cache_name: str | None = None,
                      scorer=None) -> pd.DataFrame:
    """
    HU: Mondatszintű előrejelzés MEGTARTVA (nem aggregálva) — az aspektus-elemzéshez
        (aspects.py): így egy aspektus a SAJÁT mondatának szentimentjét kapja, nem a
        teljes véleményét. Ugyanazt a cache-t használja, mint a mondatszintű címkézés
        (pl. cache_name="B_sentence"), így nem kell újra számolni.
    EN: Sentence-level predictions KEPT (not aggregated) — for the aspect analysis
        (aspects.py): an aspect gets the sentiment of ITS OWN sentence, not of the
        whole review. Uses the same cache as sentence-level labelling
        (e.g. cache_name="B_sentence"), so nothing is recomputed.
    """
    from .preprocessing import explode_to_sentences

    scorer = scorer or TransformerScorer(model_name_or_path)
    sentences = explode_to_sentences(df, text_col=text_col)
    probs = run_chunked(sentences["sentence"], scorer.predict_proba,
                        f"{cache_name}_sent" if cache_name else None)
    frame = probs_to_frame(probs, sentences.index, prefix)
    return pd.concat([sentences, frame], axis=1)


def label_approach_a(df: pd.DataFrame, text_col: str = "text_bert", granularity: str = "review",
                     three_class_model: str = CFG.zero_shot_3class_model,
                     stars_model: str = CFG.zero_shot_stars_model,
                     use_cache: bool = True, scorers: dict | None = None) -> pd.DataFrame:
    """
    HU: (A) ZERO-SHOT címkézés két kész modellel:
          1) 3 osztályos modell  -> A_label, A_p_*, A_confidence
          2) 1–5 csillagos modell -> A_score10 (a konzulens által kért 1–10-es egész pontszám)
        A 3 osztályos valószínűségből számolt pontszám is megmarad (A_score10_polarity),
        valamint diagnosztikaként a csillagokból képzett 3 osztály (A_star_label).
    EN: (A) ZERO-SHOT labelling with two ready models:
          1) 3-class model   -> A_label, A_p_*, A_confidence
          2) 1–5 star model  -> A_score10 (the integer 1–10 score requested by the supervisor)
        The score computed from the 3-class probabilities is kept too
        (A_score10_polarity), and as a diagnostic the 3 classes from the stars (A_star_label).
    """
    scorers = scorers or {}
    cache = (lambda name: f"A_{name}_{granularity}") if use_cache else (lambda name: None)

    three = scorers.get("3class") or TransformerScorer(three_class_model)
    probs, index, extra = _predict_reviews(df, three, text_col, granularity, cache("3class"))
    out = probs_to_frame(probs, index, "A")
    out = out.rename(columns={"A_score_cont": "A_score_cont_polarity",
                              "A_score10": "A_score10_polarity"})
    if extra is not None:
        out["A_is_mixed"] = extra["is_mixed"].reindex(index).to_numpy()

    stars = scorers.get("stars") or TransformerScorer(stars_model)
    star_probs = run_chunked(df[text_col].fillna(""), stars.predict_proba,
                             cache("stars") if use_cache else None)
    star_frame = pd.DataFrame(star_probs, index=pd.Index(df["review_id"]),
                              columns=[f"A_p_{k}star" for k in range(1, 6)])
    expected, cont, integer = stars_to_score(star_probs)
    star_frame["A_expected_stars"] = expected
    star_frame["A_score_cont"] = cont
    star_frame["A_score10"] = integer
    star_frame["A_star_label"] = stars_to_3class(expected)

    out = out.join(star_frame, how="left")
    out.index.name = "review_id"
    log.info("(A) label distribution: %s", out["A_label"].value_counts(normalize=True).round(3).to_dict())
    return out


def label_approach_b(df: pd.DataFrame, model_dir: str | Path | None = None,
                     text_col: str = "text_bert", granularity: str = CFG.hrast_granularity,
                     use_cache: bool = True, scorer=None) -> pd.DataFrame:
    """
    HU: (B) A HRAST-on finomhangolt modell alkalmazása (alapból mondatszinten,
        mert a HRAST mondatszintű; utána véleményszintre aggregálunk).
        A modellt előbb a bert_training.train_approach_b() tanítja.
    EN: (B) Apply the model fine-tuned on HRAST (sentence level by default,
        because HRAST is sentence-level; then aggregated to review level).
        The model is trained first by bert_training.train_approach_b().
    """
    model_dir = Path(model_dir or PATHS.models / "approach_b_hrast")
    if scorer is None and not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found — run bert_training.train_approach_b() first.")
    return label_with_classifier(df, str(model_dir), "B", text_col, granularity,
                                 f"B_{granularity}" if use_cache else None, scorer=scorer)


def label_approach_c(df: pd.DataFrame, model_dir: str | Path | None = None,
                     text_col: str = "text_bert", use_cache: bool = True, scorer=None) -> pd.DataFrame:
    """
    HU: (C) A Booking pseudo-címkéin finomhangolt modell alkalmazása (véleményszinten,
        mert véleményeken tanult). A modellt a bert_training.train_approach_c() tanítja.
    EN: (C) Apply the model fine-tuned on Booking pseudo-labels (review level, as it
        was trained on reviews). Trained by bert_training.train_approach_c().
    """
    model_dir = Path(model_dir or PATHS.models / "approach_c_self_training")
    if scorer is None and not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found — run bert_training.train_approach_c() first.")
    out = label_with_classifier(df, str(model_dir), "C", text_col, "review",
                                "C_review" if use_cache else None, scorer=scorer)
    # HU: Jelöljük, mely véleményeken tanult a (C) — ezeket az összevetésből kizárjuk.
    # EN: Flag the reviews (C) was trained on — they are excluded from the comparison.
    ids_file = model_dir / "train_review_ids.csv"
    trained_on = set(pd.read_csv(ids_file)["review_id"]) if ids_file.exists() else set()
    out["C_in_training"] = out.index.isin(trained_on)
    return out


# ==========================================================================
# 4) PSEUDO-CÍMKÉK (C) ÉS A VÉGSŐ CÍMKE / PSEUDO-LABELS (C) AND THE FINAL LABEL
# ==========================================================================
def apply_neutral_band(labels: pd.DataFrame, prefix: str,
                       band: float = CFG.review_neutral_band) -> pd.DataFrame:
    """
    HU: Véleményszintű címke a "semleges sáv" szabállyal (lásd config.review_neutral_band):
          polaritás = P(poz) − P(neg);  |polaritás| < sáv -> neutral,  különben előjel szerint.
        Az eredeti argmax-címke <prefix>_label_argmax néven megmarad.
        A <prefix>_confidence itt azt méri, mennyire mélyen van a vélemény a saját
        tartományában (0 = a határon, 1 = a tartomány közepén / szélén) — így a
        pseudo-címke kiválasztás (C) ugyanúgy működik, mint argmax esetén.
    EN: Review-level label with the "neutral band" rule (see config.review_neutral_band):
          polarity = P(pos) − P(neg);  |polarity| < band -> neutral,  otherwise by sign.
        The original argmax label is kept as <prefix>_label_argmax.
        <prefix>_confidence here measures how deep the review is inside its own region
        (0 = on the boundary, 1 = at the centre / edge) — so pseudo-label selection (C)
        works the same way as with argmax.
    """
    out = labels.copy()
    probs = out[[f"{prefix}_{s}" for s in PROB_SUFFIXES]].to_numpy()
    polarity = polarity_from_probs(probs)
    if f"{prefix}_label_argmax" not in out.columns:
        out[f"{prefix}_label_argmax"] = out[f"{prefix}_label"]
    out[f"{prefix}_label"] = np.where(np.abs(polarity) < band, "neutral",
                                      np.where(polarity > 0, "positive", "negative"))
    safe_band = max(band, 1e-9)    # HU/EN: band = 0 -> no neutral, avoid division by zero
    out[f"{prefix}_confidence"] = np.where(np.abs(polarity) < band,
                                           1 - np.abs(polarity) / safe_band,
                                           (np.abs(polarity) - band) / (1 - band))
    out[f"{prefix}_neutral_band"] = band
    log.info("(%s) neutral band ±%.2f -> %s", prefix, band,
             out[f"{prefix}_label"].value_counts(normalize=True).round(3).to_dict())
    return out


def neutral_band_table(labels: pd.DataFrame, prefix: str,
                       bands: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """
    HU: Osztályeloszlás különböző sávszélességekkel (0,0 = sima előjel, semleges nincs).
        A dolgozatba érzékenységvizsgálatként; a végső értéket a gold minta dönti el.
    EN: Class distribution for several band widths (0.0 = plain sign, no neutral).
        For the thesis as a sensitivity analysis; the final value is set by the gold sample.
    """
    probs = labels[[f"{prefix}_{s}" for s in PROB_SUFFIXES]].to_numpy()
    polarity = polarity_from_probs(probs)
    rows = {}
    for band in bands:
        lab = np.where(np.abs(polarity) < band, "neutral", np.where(polarity > 0, "positive", "negative"))
        rows[f"±{band:.1f}"] = pd.Series(lab).value_counts(normalize=True).reindex(LABELS, fill_value=0)
    out = pd.DataFrame(rows).T.round(4)
    out.index.name = "neutral band"
    return out


def select_pseudo_labels(
    labels: pd.DataFrame,
    teachers: tuple[str, ...] = ("A", "B"),
    min_confidence: float = CFG.pseudo_label_min_confidence,
    require_agreement: bool = CFG.pseudo_label_require_agreement,
    max_per_class: int | None = CFG.pseudo_label_max_per_class,
    seed: int = CFG.random_seed,
) -> pd.DataFrame:
    """
    HU: A (C) tanítóhalmazának kiválasztása (önképzés / pseudo-labelling).
          1) csak az elérhető tanár-megközelítések (A, B) oszlopait nézzük;
          2) minden tanár legalább min_confidence biztonsággal döntött;
          3) (opcionálisan) a tanárok egyetértenek;
          4) osztályonként legfeljebb max_per_class példa (kiegyensúlyozás).
        A kimaradt — bizonytalan — vélemények címkéjét a (C) modell fogja megadni.
    EN: Selecting the training set of (C) (self-training / pseudo-labelling).
          1) only the columns of the available teacher approaches (A, B) are used;
          2) every teacher decided with at least min_confidence;
          3) (optionally) the teachers agree;
          4) at most max_per_class examples per class (balancing).
        The (C) model will label the left-out — uncertain — reviews.

    Returns: review_id | pseudo_label | pseudo_confidence
    """
    available = [t for t in teachers if f"{t}_label" in labels.columns]
    if not available:
        raise ValueError(f"None of the teachers {teachers} is present in the table.")
    log.info("Pseudo-label teachers: %s", available)

    mask = pd.Series(True, index=labels.index)
    for t in available:
        mask &= labels[f"{t}_confidence"] >= min_confidence
    if require_agreement and len(available) > 1:
        first = labels[f"{available[0]}_label"]
        for t in available[1:]:
            mask &= labels[f"{t}_label"] == first

    chosen = labels.loc[mask]
    out = pd.DataFrame({
        "pseudo_label": chosen[f"{available[0]}_label"],
        "pseudo_confidence": chosen[[f"{t}_confidence" for t in available]].min(axis=1),
    })
    counts = out["pseudo_label"].value_counts().reindex(LABELS, fill_value=0)
    log.info("Pseudo-labels: %d of %d reviews (%.1f%%) | %s", len(out), len(labels),
             100 * len(out) / max(len(labels), 1), counts.to_dict())
    # HU: Ha egy osztályból alig van példa, a (C) azt az osztályt nem tudja megtanulni.
    # EN: If a class has hardly any examples, (C) cannot learn that class.
    for label, n in counts.items():
        if n < 200:
            log.warning("Only %d pseudo-labelled '%s' examples — (C) will hardly learn this class. "
                        "Consider apply_neutral_band() on the teachers or a lower min_confidence.",
                        n, label)

    if max_per_class:
        # HU: Véletlen keverés, majd csoportonként az első N — NEM groupby().apply(),
        #     mert pandas 3.0-tól az apply elhagyja a csoportosító oszlopot.
        # EN: Random shuffle, then the first N per group — NOT groupby().apply(),
        #     because from pandas 3.0 apply drops the grouping column.
        out = (out.sample(frac=1.0, random_state=seed)
                  .groupby("pseudo_label", sort=False)
                  .head(max_per_class))
        log.info("After capping at %d per class: %s", max_per_class,
                 out["pseudo_label"].value_counts().to_dict())
    return out


def combine_approaches(labels: pd.DataFrame, approaches: tuple[str, ...] = ("A", "B", "C"),
                       tie_breaker: str = CFG.tie_breaker_approach) -> pd.DataFrame:
    """
    HU: A végső címke és pontszám képzése az elérhető megközelítésekből.
          label_final    — többségi szavazat; döntetlennél a tie_breaker dönt
          n_agree        — hány megközelítés adta a végső címkét
          unanimous      — mindegyik egyetért-e (ezek a "legbiztosabb" címkék)
          score10_final  — a folytonos pontszámok átlaga, egészre kerekítve
    EN: Forming the final label and score from the available approaches.
          label_final    — majority vote; on a tie the tie_breaker decides
          n_agree        — how many approaches gave the final label
          unanimous      — do all agree (these are the "most reliable" labels)
          score10_final  — mean of the continuous scores, rounded to an integer
    """
    available = [a for a in approaches if f"{a}_label" in labels.columns]
    if not available:
        raise ValueError("No approach columns found.")
    tie = tie_breaker if tie_breaker in available else available[0]
    log.info("Combining approaches: %s (tie-breaker: %s)", available, tie)

    votes = labels[[f"{a}_label" for a in available]]

    def _vote(row) -> str:
        counts = row.value_counts()
        top = counts[counts == counts.max()].index
        return top[0] if len(top) == 1 else row[f"{tie}_label"]

    out = labels.copy()
    out["label_final"] = votes.apply(_vote, axis=1)
    out["n_agree"] = votes.eq(out["label_final"], axis=0).sum(axis=1)
    out["n_approaches"] = len(available)
    out["unanimous"] = out["n_agree"] == len(available)

    score_cols = [f"{a}_score_cont" for a in available if f"{a}_score_cont" in labels.columns]
    out["score_cont_final"] = labels[score_cols].mean(axis=1)
    out["score10_final"] = np.clip(round_half_up(out["score_cont_final"]), 1, 10).astype(int)
    return out


def save_labels(labels: pd.DataFrame, name: str) -> Path:
    """HU: Címketábla mentése a data/labels/ mappába. / EN: Save a label table to data/labels/."""
    from .data_loading import write_frame
    PATHS.ensure()
    path = write_frame(labels.reset_index(), PATHS.labels / name)
    log.info("Labels saved: %s (%d rows)", path, len(labels))
    return path


def load_labels(name: str) -> pd.DataFrame:
    from .data_loading import read_frame
    return read_frame(PATHS.labels / name).set_index("review_id")
