"""
bert_training.py — BERT finomhangolása a (B) és (C) címkéző megközelítéshez.
                   Fine-tuning BERT for labelling approaches (B) and (C).

HU: Mindkét megközelítés UGYANAZT a tanítófüggvényt használja
    (fine_tune_classifier), csak a tanítóadat más:
      (B) HRAST — kézzel annotált mondatok (emberi címke, más forrás)
      (C) Booking — az (A)/(B) legbiztosabb címkéi (gépi "ezüst" címke, saját domén)
EN: Both approaches use THE SAME training function (fine_tune_classifier),
    only the training data differs:
      (B) HRAST — human-annotated sentences (human label, different source)
      (C) Booking — the most confident labels of (A)/(B) (machine "silver" label, own domain)

HU: Futtatás: Google Colab T4 GPU (ajánlott) vagy saját 4 GB-os GPU kis batch-csel
    és gradiens-akkumulációval (a config.py-ban).
EN: Run on: Google Colab T4 GPU (recommended) or the own 4 GB GPU with a small
    batch and gradient accumulation (in config.py).
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, PATHS

log = logging.getLogger(__name__)

LABELS: list[str] = list(CFG.labels)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


# ==========================================================================
# 1) SEGÉDEK / HELPERS
# ==========================================================================
def training_batch_settings() -> tuple[int, int]:
    """
    HU: (batch, gradiens-akkumuláció) a GPU memóriája alapján. Kis GPU-n
        4 × 8 = 32 effektív batch — matematikailag ugyanaz, csak több lépésben.
    EN: (batch, gradient accumulation) based on GPU memory. On a small GPU
        4 × 8 = 32 effective batch — mathematically the same, in more steps.
    """
    import torch
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < CFG.small_gpu_vram_gb:
            return CFG.local_batch_size, CFG.local_grad_accum
    return CFG.batch_size, 1


def class_weight_vector(labels: pd.Series) -> np.ndarray:
    """
    HU: Osztálysúlyok a veszteségfüggvényhez:  w_c = n / (k · n_c)  (sklearn "balanced").
        A ritkább osztály hibája arányosan nagyobb büntetést kap.
    EN: Class weights for the loss function:  w_c = n / (k · n_c)  (sklearn "balanced").
        Errors on the rarer class get a proportionally larger penalty.
    """
    counts = labels.astype(str).value_counts().reindex(LABELS, fill_value=0)
    n, k = counts.sum(), int((counts > 0).sum())
    return np.array([n / (k * c) if c > 0 else 0.0 for c in counts], dtype=np.float32)


class _EncodedDataset:
    """
    HU: Minimális torch-kompatibilis adathalmaz (nem kell hozzá a `datasets` csomag).
    EN: Minimal torch-compatible dataset (no need for the `datasets` package).
    """

    def __init__(self, encodings: dict, labels: list[int] | None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, i: int) -> dict:
        item = {key: values[i] for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def _make_training_args(**kwargs):
    """
    HU: Verziófüggetlen TrainingArguments: a régi transformers 'evaluation_strategy'-t,
        az új 'eval_strategy'-t vár.
    EN: Version-independent TrainingArguments: old transformers expect
        'evaluation_strategy', new ones 'eval_strategy'.
    """
    from transformers import TrainingArguments
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # HU: Az adott verzióban már nem létező paramétereket figyelmeztetéssel elhagyjuk.
    # EN: Parameters that no longer exist in the installed version are dropped with a warning.
    unknown = [k for k in kwargs if k not in params]
    if unknown:
        log.warning("TrainingArguments: unsupported parameters dropped: %s", unknown)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


def _metrics_fn(eval_pred) -> dict:
    """HU: Ugyanazok a metrikák, mint mindenhol (evaluation.py). / EN: The same metrics as everywhere (evaluation.py)."""
    from .evaluation import compute_metrics
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics([ID2LABEL[int(i)] for i in labels], [ID2LABEL[int(i)] for i in preds])
    return {k: m[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")}


# ==========================================================================
# 2) KÖZÖS FINOMHANGOLÓ FÜGGVÉNY / SHARED FINE-TUNING FUNCTION
# ==========================================================================
def fine_tune_classifier(
    train_texts: pd.Series,
    train_labels: pd.Series,
    val_texts: pd.Series,
    val_labels: pd.Series,
    output_dir: str | Path,
    base_model: str = "bert-base-uncased",
    max_length: int = CFG.max_length,
    num_epochs: int = CFG.num_epochs,
    use_class_weights: bool = True,
) -> dict:
    """
    HU: 3 osztályos szekvencia-osztályozó finomhangolása, súlyozott veszteséggel.
        A legjobb (validációs makro-F1 szerinti) modell kerül mentésre az
        output_dir mappába, mellé a training_info.json.
    EN: Fine-tune a 3-class sequence classifier with a weighted loss.
        The best model (by validation macro-F1) is saved to output_dir, together
        with training_info.json.

    Returns: {"model_dir", "train_minutes", "val_metrics", "log_history"}
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, accum = training_batch_settings()
    log.info("Fine-tuning %s | train %d | val %d | batch %d × accum %d",
             base_model, len(train_texts), len(val_texts), batch, accum)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def encode(texts: pd.Series, labels: pd.Series) -> _EncodedDataset:
        # HU: Kitöltés NÉLKÜL tokenizálunk; a collator kötegenként tölt ki.
        # EN: Tokenise WITHOUT padding; the collator pads per batch.
        enc = tokenizer(texts.astype(str).tolist(), truncation=True, max_length=max_length)
        return _EncodedDataset(dict(enc), labels.astype(str).map(LABEL2ID).astype(int).tolist())

    ds_train, ds_val = encode(train_texts, train_labels), encode(val_texts, val_labels)

    weights = torch.tensor(class_weight_vector(train_labels) if use_class_weights
                           else np.ones(len(LABELS), dtype=np.float32))
    log.info("Class weights: %s", dict(zip(LABELS, weights.numpy().round(3))))

    class WeightedTrainer(Trainer):
        """HU: Trainer súlyozott keresztentrópiával. / EN: Trainer with weighted cross-entropy."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits.view(-1, len(LABELS)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = _make_training_args(
        output_dir=str(output_dir / "checkpoints"),
        seed=CFG.random_seed,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=batch * 4,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        # HU: A transformers v5 már nem ismeri a warmup_ratio-t -> lépésszámra váltjuk.
        # EN: transformers v5 no longer knows warmup_ratio -> converted to a number of steps.
        warmup_steps=int(CFG.warmup_ratio * num_epochs * np.ceil(len(train_texts) / (batch * accum))),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer_kwargs = dict(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                          compute_metrics=_metrics_fn)
    # HU: Újabb transformers: 'processing_class', régebbi: 'tokenizer'.
    # EN: Newer transformers: 'processing_class', older: 'tokenizer'.
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)
    # HU: A saját veszteségfüggvény nem használja a num_items_in_batch-et, ezért a Trainer
    #     maga ossza el a veszteséget a gradiens-akkumulációval (különben a naplózott
    #     tanítási veszteség accum-szorosa lenne).
    # EN: The custom loss does not use num_items_in_batch, so the Trainer itself must
    #     divide the loss by the gradient accumulation (otherwise the logged training
    #     loss would be accum times too large).
    trainer.model_accepts_loss_kwargs = False

    started = time.time()
    trainer.train()
    train_minutes = (time.time() - started) / 60

    # HU: A legjobb ellenőrzőpont újratöltése from_pretrained-del. A Trainer saját
    #     visszatöltése a régi BERT-súlyneveknél (LayerNorm gamma/beta vs weight/bias)
    #     kihagyhat paramétereket — a from_pretrained ezeket helyesen átnevezi.
    # EN: Reload the best checkpoint with from_pretrained. The Trainer's own reload can
    #     skip parameters with the legacy BERT weight names (LayerNorm gamma/beta vs
    #     weight/bias) — from_pretrained renames them correctly.
    best = trainer.state.best_model_checkpoint
    if best:
        log.info("Reloading best checkpoint: %s", Path(best).name)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(best).to(trainer.args.device)
    val_metrics = {k.replace("eval_", ""): float(v) for k, v in trainer.evaluate().items()
                   if isinstance(v, (int, float))}

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    info = {"base_model": base_model, "n_train": int(len(train_texts)), "n_val": int(len(val_texts)),
            "train_minutes": round(train_minutes, 2), "val_metrics": val_metrics,
            "batch": batch, "grad_accum": accum, "epochs": num_epochs,
            "max_length": max_length, "seed": CFG.random_seed}
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    log.info("Model saved: %s (%.1f min) | val macro-F1 = %.4f",
             output_dir, train_minutes, val_metrics.get("f1_macro", float("nan")))
    return {"model_dir": output_dir, "train_minutes": train_minutes,
            "val_metrics": val_metrics, "log_history": trainer.state.log_history}


# ==========================================================================
# 3) (B) HRAST / (C) ÖNKÉPZÉS / (C) SELF-TRAINING
# ==========================================================================
def split_hrast(sentences: pd.DataFrame, val_size: float = CFG.hrast_val_size,
                test_size: float = CFG.hrast_test_size, seed: int = CFG.random_seed):
    """
    HU: HRAST felosztása train/val/test részre. Ha van vélemény-azonosító (group_id),
        egy vélemény összes mondata UGYANABBA a részbe kerül (nincs adatszivárgás).
    EN: Split HRAST into train/val/test. If there is a review id (group_id), all
        sentences of one review go into THE SAME part (no data leakage).
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    has_groups = sentences["group_id"].notna().all()
    if has_groups:
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        rest_idx, test_idx = next(outer.split(sentences, groups=sentences["group_id"]))
        rest, test = sentences.iloc[rest_idx], sentences.iloc[test_idx]
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        tr_idx, va_idx = next(inner.split(rest, groups=rest["group_id"]))
        train, val = rest.iloc[tr_idx], rest.iloc[va_idx]
    else:
        rest, test = train_test_split(sentences, test_size=test_size, random_state=seed,
                                      stratify=sentences["label"])
        train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                      random_state=seed, stratify=rest["label"])
    log.info("HRAST split (%s) — train %d | val %d | test %d",
             "grouped by review" if has_groups else "stratified", len(train), len(val), len(test))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_approach_b(hrast_sentences: pd.DataFrame, output_dir: str | Path | None = None,
                     base_model: str = CFG.hrast_base_model) -> dict:
    """
    HU: (B) BERT finomhangolása a HRAST-on. A HRAST teszthalmazán mért eredmény
        az EGYETLEN, emberi címkén mért teljesítmény a projektben (a gold mintán kívül).
    EN: (B) Fine-tune BERT on HRAST. The result on the HRAST test set is the ONLY
        performance measured on human labels in the project (besides the gold sample).

    Returns: fine_tune_classifier() result + "test_metrics", "test_predictions"
    """
    from .bert_labelling import TransformerScorer
    from .evaluation import compute_metrics

    output_dir = Path(output_dir or PATHS.models / "approach_b_hrast")
    train, val, test = split_hrast(hrast_sentences)
    result = fine_tune_classifier(train["sentence"], train["label"], val["sentence"], val["label"],
                                  output_dir, base_model=base_model)

    scorer = TransformerScorer(str(output_dir))
    probs = scorer.predict_proba(test["sentence"].tolist())
    test = test.assign(predicted=np.array(LABELS)[probs.argmax(axis=1)])
    result["test_metrics"] = compute_metrics(test["label"], test["predicted"])
    result["test_predictions"] = test
    log.info("(B) HRAST test macro-F1 = %.4f", result["test_metrics"]["f1_macro"])
    return result


def train_approach_c(prepared: pd.DataFrame, pseudo: pd.DataFrame,
                     output_dir: str | Path | None = None, text_col: str = "text_bert",
                     base_model: str = CFG.self_training_base_model,
                     val_share: float = 0.1) -> dict:
    """
    HU: (C) BERT finomhangolása a Booking pseudo-címkéin (select_pseudo_labels kimenete).
        FIGYELEM: a validációs makro-F1 itt a TANÁROKKAL való egyezést méri, nem a
        valódi pontosságot — a valódi minőséget a gold mintán kell mérni.
    EN: (C) Fine-tune BERT on the Booking pseudo-labels (output of select_pseudo_labels).
        WARNING: the validation macro-F1 here measures agreement WITH THE TEACHERS,
        not true accuracy — true quality has to be measured on the gold sample.
    """
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir or PATHS.models / "approach_c_self_training")
    data = (prepared.set_index("review_id")[[text_col]]
                    .join(pseudo[["pseudo_label"]], how="inner")
                    .reset_index())
    train, val = train_test_split(data, test_size=val_share, random_state=CFG.random_seed,
                                  stratify=data["pseudo_label"])
    # HU: A tanításban részt vett vélemények listája. Ezeken a (C) a tanárokat
    #     "visszamondja", ezért az összevetésből ki kell őket zárni.
    # EN: List of the reviews used in training. On these (C) "repeats" the teachers,
    #     so they must be excluded from the comparison.
    output_dir.mkdir(parents=True, exist_ok=True)
    train[["review_id"]].to_csv(output_dir / "train_review_ids.csv", index=False)
    return fine_tune_classifier(train[text_col], train["pseudo_label"],
                                val[text_col], val["pseudo_label"],
                                output_dir, base_model=base_model)
