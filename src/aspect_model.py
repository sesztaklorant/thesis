"""
aspect_model.py — 2/a réteg: tanult, többcímkés aspektus-osztályozó (HRAST).
                  Layer 2/a: learned multi-label aspect classifier (HRAST).

HU: Az első réteg megmondja, MILYEN egy mondat (negatív / semleges / pozitív). Ez a réteg
    azt mondja meg, MIRŐL szól: szoba, személyzet, fürdőszoba, zaj, wifi... Egy mondat
    egyszerre több aspektushoz is tartozhat ("a szoba tiszta, de zajos"), ezért ez
    TÖBBCÍMKÉS (multi-label) feladat: aspektusonként egy-egy igen/nem döntés, szigmoiddal.

    Miért kell, ha már van kulcsszólista (aspects.py)?
      • a kulcsszólista csak a felsorolt szavakat ismeri ("we couldn't sleep" -> zaj? nem);
      • a tanult modell a kontextusból általánosít, és EMBERI címkén mérhető (HRAST-teszt);
      • a kulcsszólista megmarad ALAPVONALNAK: a kettő összevetése számszerű eredmény.

EN: The first layer says WHAT a sentence is like (negative / neutral / positive). This layer
    says WHAT IT IS ABOUT: room, staff, bathroom, noise, wifi... A sentence can belong to
    several aspects at once ("the room is clean but noisy"), so this is a MULTI-LABEL task:
    one yes/no decision per aspect, with a sigmoid.

    Why is it needed when there is already a keyword list (aspects.py)?
      • the keyword list only knows the listed words ("we couldn't sleep" -> noise? no);
      • the learned model generalises from context and is measurable on HUMAN labels;
      • the keyword list stays as the BASELINE: comparing the two is a quantitative result.
"""

from __future__ import annotations

import html
import inspect
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, HRAST_ASPECT_MAP, PATHS

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = PATHS.models / "aspect_model_hrast"
PROB_PREFIX = "aprob_"     # HU: aspektus-valószínűség / EN: aspect probability
FLAG_PREFIX = "asp_"       # HU: aspektus 0/1 a küszöb után / EN: aspect 0/1 after the threshold


# ==========================================================================
# 1) ADAT-ELŐKÉSZÍTÉS / DATA PREPARATION
# ==========================================================================
def prepare_hrast_aspects(raw: pd.DataFrame, aspect_map: dict[str, str] | None = None,
                          text_col: str | None = None,
                          min_examples: int = CFG.aspect_min_examples) -> tuple[pd.DataFrame, list[str]]:
    """
    HU: HRAST -> mondatonként egy sor, aspektusonként egy 0/1 oszlop (egységes angol névvel).
        • az ismétlődő mondatoknál az aspektusok UNIÓJA marad (ha bárhol 1, akkor 1);
        • a min_examples-nél ritkább aspektus kimarad (nem tanulható megbízhatóan);
        • az aspektus nélküli mondatok MARADNAK: a modellnek a "nincs aspektus" esetet is
          meg kell tanulnia.
    EN: HRAST -> one row per sentence, one 0/1 column per aspect (uniform English name).
        • for repeated sentences the UNION of aspects is kept (1 if 1 anywhere);
        • aspects rarer than min_examples are dropped (cannot be learned reliably);
        • sentences without any aspect STAY: the model must learn the "no aspect" case too.

    Returns: (table: sentence + aspect columns, list of aspect names)
    """
    from .data_loading import _HRAST_ALIASES, resolve_columns

    aspect_map = aspect_map or HRAST_ASPECT_MAP
    text_col = text_col or resolve_columns(raw.columns, {"text": _HRAST_ALIASES["text"]}).get("text")
    if text_col is None:
        raise KeyError(f"Text column not found. HRAST columns: {list(raw.columns)}")

    lowered = {str(c).strip().lower(): c for c in raw.columns}
    found = {name: lowered[src.lower()] for src, name in aspect_map.items() if src.lower() in lowered}
    missing = sorted(set(aspect_map.values()) - set(found))
    if missing:
        log.warning("HRAST aspect columns not found (skipped): %s", missing)
    if not found:
        raise KeyError(f"No aspect column recognised. HRAST columns: {list(raw.columns)}")

    data = pd.DataFrame({"sentence": raw[text_col].astype(str).map(html.unescape).str.strip()})
    for name, col in found.items():
        data[name] = (pd.to_numeric(raw[col], errors="coerce").fillna(0) > 0).astype(int)
    data = data[data["sentence"].str.len() > 0]
    data = data.groupby("sentence", sort=False).max().reset_index()

    counts = data[list(found)].sum().sort_values(ascending=False)
    keep = [name for name in counts.index if counts[name] >= min_examples]
    dropped = [name for name in counts.index if counts[name] < min_examples]
    if dropped:
        log.warning("Aspects with fewer than %d examples dropped: %s", min_examples,
                    {n: int(counts[n]) for n in dropped})
    log.info("HRAST aspect data: %d sentences, %d aspects | without aspect: %.1f%%",
             len(data), len(keep), 100 * (data[keep].sum(axis=1) == 0).mean())
    return data[["sentence", *keep]].reset_index(drop=True), keep


def split_aspect_data(data: pd.DataFrame, val_size: float = CFG.aspect_val_size,
                      test_size: float = CFG.aspect_test_size, seed: int = CFG.random_seed):
    """
    HU: Véletlen felosztás train/val/test részre (a HRAST-ban nincs vélemény-azonosító).
    EN: Random split into train/val/test (HRAST has no review id).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    n_test, n_val = int(round(test_size * len(data))), int(round(val_size * len(data)))
    test, val, train = (data.iloc[order[:n_test]], data.iloc[order[n_test:n_test + n_val]],
                        data.iloc[order[n_test + n_val:]])
    log.info("Aspect split — train %d | val %d | test %d", len(train), len(val), len(test))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def positive_weights(y: np.ndarray, cap: float = CFG.aspect_max_pos_weight) -> np.ndarray:
    """
    HU: Aspektusonkénti súly a pozitív (=előfordul) példákra: negatív/pozitív arány, felülről
        korlátozva. A ritka aspektusok (pl. lift) így nem "tűnnek el" a tanításban.
    EN: Per-aspect weight for positive (=present) examples: negative/positive ratio, capped
        from above. Rare aspects (e.g. lift) thus do not "disappear" during training.
    """
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return np.clip(neg / np.maximum(pos, 1), 1.0, cap).astype(np.float32)


# ==========================================================================
# 2) KÜSZÖBÖK ÉS METRIKÁK / THRESHOLDS AND METRICS
# ==========================================================================
def tune_thresholds(y_true: np.ndarray, probs: np.ndarray,
                    grid: np.ndarray | None = None) -> np.ndarray:
    """
    HU: Aspektusonként az a küszöb, amely a VALIDÁCIÓS halmazon a legnagyobb F1-et adja.
        A ritka aspektusoknál a 0,5 gyakran túl magas — ezért nem egy közös küszöb van.
        (A teszthalmazt itt NEM használjuk, különben a mért eredmény optimista lenne.)
    EN: Per aspect, the threshold giving the highest F1 on the VALIDATION set. For rare
        aspects 0.5 is often too high — hence no single shared threshold.
        (The test set is NOT used here, otherwise the measured result would be optimistic.)
    """
    from sklearn.metrics import f1_score

    grid = np.arange(0.10, 0.91, 0.05) if grid is None else grid
    thresholds = np.full(y_true.shape[1], 0.5)
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores = [f1_score(y_true[:, k], probs[:, k] >= t, zero_division=0) for t in grid]
        thresholds[k] = float(grid[int(np.argmax(scores))])
    return thresholds


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    HU: Aspektusonkénti precision / recall / F1 / support, plusz mikro- és makro-átlag.
          mikro — minden döntés egyformán számít (a gyakori aspektusok dominálnak);
          makro — minden aspektus egyformán számít (a ritkák is látszanak).
    EN: Per-aspect precision / recall / F1 / support, plus micro and macro averages.
          micro — every decision counts equally (frequent aspects dominate);
          macro — every aspect counts equally (rare ones are visible too).
    """
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    table = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=names)
    for avg in ("micro", "macro"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        table.loc[f"{avg} avg"] = [pa, ra, fa, int(y_true.sum())]
    table["support"] = table["support"].astype(int)
    return table.round(4)


def keyword_predictions(sentences: pd.Series, names: list[str]) -> pd.DataFrame:
    """
    HU: A kulcsszavas alapvonal (aspects.py) jóslatai ugyanazokra az aspektusnevekre.
        Ahol nincs kulcsszólista (pl. comfort, view), az oszlop NaN.
    EN: Predictions of the keyword baseline (aspects.py) for the same aspect names.
        Where there is no keyword list (e.g. comfort, view), the column is NaN.
    """
    from .aspects import ASPECT_KEYWORDS, detect_aspects

    found = sentences.map(detect_aspects)
    out = pd.DataFrame(index=sentences.index)
    for name in names:
        out[name] = found.map(lambda lst, n=name: int(n in lst)) if name in ASPECT_KEYWORDS else np.nan
    return out


def compare_with_keywords(y_true: np.ndarray, model_pred: np.ndarray, keyword_pred: pd.DataFrame,
                          names: list[str]) -> pd.DataFrame:
    """
    HU: Tanult modell vs. kulcsszólista aspektusonként, a HRAST TESZThalmazán (emberi címke).
        Csak azokon az aspektusokon, amelyekre van kulcsszólista.
    EN: Learned model vs keyword list per aspect, on the HRAST TEST set (human labels).
        Only on the aspects that have a keyword list.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    rows = []
    for k, name in enumerate(names):
        if keyword_pred[name].isna().all():
            continue
        kw = keyword_pred[name].to_numpy().astype(int)
        rows.append({
            "aspect": name, "support": int(y_true[:, k].sum()),
            "keyword_precision": precision_score(y_true[:, k], kw, zero_division=0),
            "keyword_recall": recall_score(y_true[:, k], kw, zero_division=0),
            "keyword_f1": f1_score(y_true[:, k], kw, zero_division=0),
            "model_precision": precision_score(y_true[:, k], model_pred[:, k], zero_division=0),
            "model_recall": recall_score(y_true[:, k], model_pred[:, k], zero_division=0),
            "model_f1": f1_score(y_true[:, k], model_pred[:, k], zero_division=0),
        })
    out = pd.DataFrame(rows).set_index("aspect")
    out["f1_gain"] = out["model_f1"] - out["keyword_f1"]
    return out.sort_values("support", ascending=False).round(4)


def plot_keyword_vs_model(comparison: pd.DataFrame, save_as: str | None = None):
    """
    HU: Páros pontdiagram: kulcsszólista vs. tanult modell F1-e aspektusonként.
    EN: Dumbbell plot: F1 of the keyword list vs the learned model per aspect.
    """
    import matplotlib.pyplot as plt
    from .config import APPROACH_COLORS, INK
    from .evaluation import save_figure

    d = comparison.sort_values("model_f1")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.5, max(3.4, 0.42 * len(d) + 2.0)))
    ax.hlines(y, d["keyword_f1"], d["model_f1"], color=INK["axis"], lw=2, zorder=1)
    ax.scatter(d["keyword_f1"], y, s=90, zorder=2, color=INK["muted"], label="keyword list (baseline)",
               edgecolor=INK["surface"], linewidth=1.5)
    ax.scatter(d["model_f1"], y, s=90, zorder=2, color=APPROACH_COLORS["B"], label="learned model (BERT)",
               edgecolor=INK["surface"], linewidth=1.5)
    ax.set_yticks(y, d.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on the HRAST test set (human labels)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Aspect detection — keyword list vs learned model", loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


# ==========================================================================
# 3) TANÍTÁS / TRAINING
# ==========================================================================
def train_aspect_model(data: pd.DataFrame, names: list[str], output_dir: str | Path | None = None,
                       base_model: str = CFG.aspect_base_model, num_epochs: int = CFG.num_epochs) -> dict:
    """
    HU: Többcímkés BERT finomhangolása a HRAST-on.
          • veszteség: BCEWithLogits, aspektusonkénti pos_weight-tel (ritka aspektusok);
          • a legjobb epoch a validációs mikro-F1 alapján;
          • küszöbök a validációs halmazon hangolva (thresholds.json);
          • végső mérés a TESZThalmazon + összevetés a kulcsszólistával.
    EN: Fine-tune a multi-label BERT on HRAST.
          • loss: BCEWithLogits with a per-aspect pos_weight (rare aspects);
          • best epoch by validation micro-F1;
          • thresholds tuned on the validation set (thresholds.json);
          • final measurement on the TEST set + comparison with the keyword list.

    Returns: {"model_dir", "train_minutes", "thresholds", "test_report", "keyword_comparison"}
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer)

    from .bert_training import _EncodedDataset, _make_training_args, training_batch_settings

    output_dir = Path(output_dir or DEFAULT_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = split_aspect_data(data)
    batch, accum = training_batch_settings()
    id2label = {i: n for i, n in enumerate(names)}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(names), id2label=id2label, label2id={n: i for i, n in id2label.items()},
        problem_type="multi_label_classification")

    def encode(part: pd.DataFrame) -> _EncodedDataset:
        enc = tokenizer(part["sentence"].tolist(), truncation=True, max_length=CFG.aspect_max_length)
        return _EncodedDataset(dict(enc), part[names].astype(np.float32).to_numpy().tolist())

    ds_train, ds_val = encode(train), encode(val)
    pos_weight = torch.tensor(positive_weights(train[names].to_numpy()))
    log.info("pos_weight per aspect: %s", dict(zip(names, pos_weight.numpy().round(2))))

    def metrics_fn(eval_pred) -> dict:
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        pred = (1 / (1 + np.exp(-logits))) >= 0.5
        return {"f1_micro": f1_score(labels, pred, average="micro", zero_division=0),
                "f1_macro": f1_score(labels, pred, average="macro", zero_division=0)}

    class MultiLabelTrainer(Trainer):
        """HU: BCE-veszteség aspektusonkénti súllyal. / EN: BCE loss with per-aspect weight."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    args = _make_training_args(
        output_dir=str(output_dir / "checkpoints"), seed=CFG.random_seed, num_train_epochs=num_epochs,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
        per_device_eval_batch_size=batch * 4, learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_steps=int(CFG.warmup_ratio * num_epochs * np.ceil(len(train) / (batch * accum))),
        fp16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="f1_micro", greater_is_better=True,
        save_total_limit=2, report_to="none")
    trainer_kwargs = dict(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=metrics_fn)
    params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    trainer = MultiLabelTrainer(**trainer_kwargs)
    trainer.model_accepts_loss_kwargs = False     # HU/EN: see bert_training.fine_tune_classifier

    started = time.time()
    trainer.train()
    train_minutes = (time.time() - started) / 60
    if trainer.state.best_model_checkpoint:
        trainer.model = AutoModelForSequenceClassification.from_pretrained(
            trainer.state.best_model_checkpoint).to(trainer.args.device)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # HU: Küszöbök a validáción, mérés a teszten. / EN: Thresholds on validation, measurement on test.
    scorer = AspectScorer(output_dir, thresholds=np.full(len(names), 0.5))
    thresholds = tune_thresholds(val[names].to_numpy(), scorer.predict_proba(val["sentence"].tolist()))
    (output_dir / "thresholds.json").write_text(json.dumps(dict(zip(names, thresholds.tolist())), indent=2))

    test_probs = scorer.predict_proba(test["sentence"].tolist())
    test_pred = (test_probs >= thresholds).astype(int)
    y_test = test[names].to_numpy()
    report = multilabel_report(y_test, test_pred, names)
    comparison = compare_with_keywords(y_test, test_pred, keyword_predictions(test["sentence"], names), names)

    info = {"base_model": base_model, "aspects": names, "n_train": len(train), "n_val": len(val),
            "n_test": len(test), "train_minutes": round(train_minutes, 2), "epochs": num_epochs,
            "test_f1_micro": float(report.loc["micro avg", "f1"]),
            "test_f1_macro": float(report.loc["macro avg", "f1"])}
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    log.info("Aspect model saved: %s (%.1f min) | test micro-F1 %.4f | macro-F1 %.4f", output_dir,
             train_minutes, info["test_f1_micro"], info["test_f1_macro"])
    return {"model_dir": output_dir, "train_minutes": train_minutes, "thresholds": thresholds,
            "test_report": report, "keyword_comparison": comparison, "test": test.assign(
                **{f"{PROB_PREFIX}{n}": test_probs[:, k] for k, n in enumerate(names)})}


# ==========================================================================
# 4) ALKALMAZÁS A BOOKING-MONDATOKRA / APPLYING TO THE BOOKING SENTENCES
# ==========================================================================
class AspectScorer:
    """
    HU: A betanított aspektusmodell kötegelt alkalmazása (szigmoid valószínűségek).
        Ugyanaz a gyorsítás, mint a TransformerScorer-nél: hossz szerinti rendezés, fp16.
    EN: Batched application of the trained aspect model (sigmoid probabilities).
        Same speed-ups as TransformerScorer: sorting by length, fp16.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, thresholds: np.ndarray | None = None,
                 batch_size: int = CFG.inference_batch_size, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from .bert_labelling import pick_device

        self.name = str(model_dir)
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.names = [self.model.config.id2label[i] for i in sorted(self.model.config.id2label, key=int)]
        if thresholds is None:
            path = Path(model_dir) / "thresholds.json"
            stored = json.loads(path.read_text()) if path.exists() else {}
            thresholds = np.array([stored.get(n, 0.5) for n in self.names])
        self.thresholds = np.asarray(thresholds, dtype=float)
        self._torch = torch

    def predict_proba(self, texts) -> np.ndarray:
        torch = self._torch
        texts = ["" if t is None else str(t) for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(self.names)), dtype=np.float32)
        for b in range(int(np.ceil(len(texts) / self.batch_size))):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                 max_length=CFG.aspect_max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(**enc).logits
                else:
                    logits = self.model(**enc).logits
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        return out


def predict_aspects(sentences: pd.DataFrame, model_dir: str | Path = DEFAULT_MODEL_DIR,
                    text_col: str = "sentence", cache_name: str | None = "aspects_sentence",
                    scorer=None) -> pd.DataFrame:
    """
    HU: Aspektusok a Booking-mondatokra (darabolt, folytatható futás, mint a címkézésnél).
        Új oszlopok: aprob_<aspektus> (valószínűség), asp_<aspektus> (0/1 a küszöb után),
        aspect_list ("room;noise"), n_aspects.
    EN: Aspects for the Booking sentences (chunked, resumable run, as in labelling).
        New columns: aprob_<aspect> (probability), asp_<aspect> (0/1 after the threshold),
        aspect_list ("room;noise"), n_aspects.
    """
    from .bert_labelling import run_chunked

    scorer = scorer or AspectScorer(model_dir)
    probs = run_chunked(sentences[text_col].fillna(""), scorer.predict_proba, cache_name)
    flags = (probs >= scorer.thresholds).astype(int)
    out = sentences.copy()
    for k, name in enumerate(scorer.names):
        out[f"{PROB_PREFIX}{name}"] = probs[:, k]
        out[f"{FLAG_PREFIX}{name}"] = flags[:, k]
    names = np.array(scorer.names)
    out["aspect_list"] = [";".join(names[row.astype(bool)]) for row in flags]
    out["n_aspects"] = flags.sum(axis=1)
    log.info("Aspects predicted for %d sentences | without aspect: %.1f%%",
             len(out), 100 * (out["n_aspects"] == 0).mean())
    return out


def aspect_names_in(frame: pd.DataFrame) -> list[str]:
    """HU: A táblában lévő aspektusnevek (asp_* oszlopokból). / EN: Aspect names present (from asp_* columns)."""
    return [c[len(FLAG_PREFIX):] for c in frame.columns if c.startswith(FLAG_PREFIX)]


def keyword_agreement(sent_df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    HU: Booking-mondatokon: mennyire egyezik a tanult modell a kulcsszólistával
        aspektusonként (egyezési arány, Cohen-kappa, gyakoriság mindkét módszerrel).
        Itt nincs emberi címke — ez a két módszer KONZISZTENCIÁJÁT méri.
    EN: On Booking sentences: how far the learned model agrees with the keyword list per
        aspect (agreement rate, Cohen's kappa, frequency by both methods). There is no
        human label here — this measures the CONSISTENCY of the two methods.
    """
    from sklearn.metrics import cohen_kappa_score

    names = aspect_names_in(sent_df)
    keywords = keyword_predictions(sent_df[text_col], names)
    rows = []
    for name in names:
        if keywords[name].isna().all():
            continue
        model, kw = sent_df[f"{FLAG_PREFIX}{name}"].to_numpy(), keywords[name].to_numpy().astype(int)
        rows.append({"aspect": name, "share_model": model.mean(), "share_keyword": kw.mean(),
                     "agreement": (model == kw).mean(),
                     "cohen_kappa": cohen_kappa_score(model, kw) if model.any() or kw.any() else np.nan})
    return pd.DataFrame(rows).set_index("aspect").sort_values("share_model", ascending=False).round(4)
