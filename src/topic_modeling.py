"""
topic_modeling.py — 2/b réteg: altémák felfedezése (BERTopic-elvű saját megvalósítás).
                    Layer 2/b: discovering subtopics (own implementation of the BERTopic principle).

HU: A 2/a réteg megmondja, hogy egy panasz a FÜRDŐSZOBÁRÓL szól. Ez a réteg azt, hogy
    PONTOSAN MI a gond vele: nincs meleg víz? gyenge a vízsugár? penészes? Ezeket az altémákat
    nem adjuk meg előre — az adatból fedezzük fel őket, négy lépésben (Grootendorst, 2022):

      1. BEÁGYAZÁS   — minden mondat egy 384 dimenziós vektor (Sentence-BERT); a hasonló
                       jelentésű mondatok vektorai közel esnek egymáshoz;
      2. DIMENZIÓCSÖKKENTÉS — UMAP 5 dimenzióra (a sűrűségalapú fürtözés sok dimenzióban rossz);
      3. FÜRTÖZÉS    — HDBSCAN: a sűrű csoportok a témák; ami sehova sem illik, az "zaj" (-1);
      4. LEÍRÁS      — c-TF-IDF: a fürt (téma) összes mondatát egy dokumentumnak tekintve
                       melyik szó / szópár jellemző RÁ a többi témához képest.

    Miért saját megvalósítás, és nem a bertopic csomag?
      • minden lépés átlátható és a dolgozatban leírható (pályamű!);
      • Windows-on a hdbscan csomag fordítást igényelhet — a scikit-learn beépített HDBSCAN-ja nem;
      • a c-TF-IDF ugyanaz a logika, mint a 04-es notebook TF-IDF-elemzése.

EN: Layer 2/a says that a complaint is about the BATHROOM. This layer says WHAT EXACTLY is
    wrong with it: no hot water? weak water pressure? mould? These subtopics are not given in
    advance — they are discovered from the data, in four steps (Grootendorst, 2022):

      1. EMBEDDING   — every sentence becomes a 384-dimensional vector (Sentence-BERT);
                       sentences with similar meaning end up close to each other;
      2. DIMENSION REDUCTION — UMAP to 5 dimensions (density clustering is poor in many dims);
      3. CLUSTERING  — HDBSCAN: dense groups are the topics; what fits nowhere is "noise" (-1);
      4. DESCRIPTION — c-TF-IDF: treating all sentences of a cluster (topic) as one document,
                       which words / word pairs characterise it compared with the other topics.

    Why an own implementation instead of the bertopic package?
      • every step is transparent and can be described in the thesis (software project!);
      • on Windows the hdbscan package may need compiling — scikit-learn's HDBSCAN does not;
      • c-TF-IDF follows the same logic as the TF-IDF analysis of notebook 04.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
arXiv:2203.05794.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:                      # HU: csak a típusjelöléshez / EN: for type hints only
    from scipy import sparse

from .config import CFG, INK, NON_THEMATIC_ASPECTS, PATHS, SENTIMENT_COLORS

log = logging.getLogger(__name__)

# HU: Stopszavak: az angol lista, DE a tagadószavak nélkül ("not working", "no hot water"
#     egy panasz lényege), plusz néhány hotel-doménbeli töltelékszó.
# EN: Stopwords: the English list, BUT without negation words ("not working", "no hot water"
#     are the essence of a complaint), plus a few hotel-domain filler words.
_NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "neither", "without", "cannot"}
_DOMAIN_STOPWORDS = {"hotel", "stay", "stayed", "staying", "also", "would", "could", "get", "got",
                     "really", "just", "one", "us", "we", "even", "bit", "little", "quite", "place"}


def stopwords() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return sorted((set(ENGLISH_STOP_WORDS) - _NEGATIONS) | _DOMAIN_STOPWORDS)


# ==========================================================================
# 1) BEÁGYAZÁS / EMBEDDING
# ==========================================================================
class SentenceEmbedder:
    """
    HU: Sentence-BERT beágyazó (sentence-transformers). A vektorok egységnyi hosszúak, így a
        koszinusz-hasonlóság egyszerű skaláris szorzat.
    EN: Sentence-BERT embedder (sentence-transformers). Vectors have unit length, so cosine
        similarity is a plain dot product.
    """

    def __init__(self, model_name: str = CFG.embedding_model, batch_size: int = CFG.embedding_batch_size,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer

        from .bert_labelling import pick_device

        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=pick_device(device))

    def embed(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def embed_sentences(texts: pd.Series, cache_name: str | None = "embeddings", embedder=None) -> np.ndarray:
    """
    HU: Beágyazások darabolt, folytatható számítása (ugyanaz a cache, mint a címkézésnél).
    EN: Chunked, resumable computation of embeddings (same cache as for labelling).
    """
    from .bert_labelling import run_chunked

    embedder = embedder or SentenceEmbedder()
    emb = run_chunked(texts.fillna("").astype(str), embedder.embed, cache_name)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


# ==========================================================================
# 2) DIMENZIÓCSÖKKENTÉS ÉS FÜRTÖZÉS / DIMENSION REDUCTION AND CLUSTERING
# ==========================================================================
def reduce_dimensions(embeddings: np.ndarray, n_components: int = CFG.umap_n_components,
                      n_neighbors: int = CFG.umap_n_neighbors, seed: int = CFG.random_seed) -> np.ndarray:
    """
    HU: UMAP (koszinusz-metrika). Ha az umap-learn nincs telepítve, SVD-re esik vissza
        figyelmeztetéssel — az csak próbafutásra jó, a végleges futáshoz UMAP kell.
    EN: UMAP (cosine metric). If umap-learn is not installed it falls back to SVD with a
        warning — fine for a trial run only, the final run needs UMAP.
    """
    n_components = min(n_components, embeddings.shape[1] - 1, max(len(embeddings) - 2, 1))
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(embeddings) - 1), n_components=n_components,
                            min_dist=0.0, metric="cosine", random_state=seed)
    except ImportError:
        from sklearn.decomposition import TruncatedSVD
        log.warning("umap-learn is not installed — falling back to TruncatedSVD (trial runs only). "
                    "pip install umap-learn")
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    return reducer.fit_transform(embeddings)


def cluster(reduced: np.ndarray, min_cluster_size: int, min_samples: int = CFG.topic_min_samples) -> np.ndarray:
    """
    HU: HDBSCAN-fürtözés (scikit-learn ≥ 1.3). -1 = zaj (egyik témába sem illik).
    EN: HDBSCAN clustering (scikit-learn ≥ 1.3). -1 = noise (fits no topic).
    """
    from sklearn.cluster import HDBSCAN

    import inspect
    kwargs = dict(min_cluster_size=max(min_cluster_size, 2), min_samples=min(min_samples, max(min_cluster_size, 2)))
    if "copy" in inspect.signature(HDBSCAN.__init__).parameters:
        kwargs["copy"] = True        # HU/EN: silences a scikit-learn FutureWarning
    model = HDBSCAN(**kwargs)
    return model.fit_predict(reduced)


def renumber_by_size(labels: np.ndarray) -> np.ndarray:
    """HU: Témák átszámozása méret szerint (0 = legnagyobb), -1 marad. / EN: Renumber topics by size (0 = largest), -1 stays."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    mapping = {old: new for new, old in enumerate(ids[np.argsort(-counts, kind="stable")])}
    return np.array([mapping.get(x, -1) for x in labels])


def reassign_outliers(embeddings: np.ndarray, labels: np.ndarray,
                      threshold: float = CFG.topic_outlier_threshold) -> np.ndarray:
    """
    HU: A zajpontokat (-1) a legközelebbi téma középpontjához soroljuk, ha a koszinusz-
        hasonlóság legalább threshold. A HDBSCAN óvatos: sok mondatot hagy zajként, ami
        üzleti elemzésnél kár lenne; a küszöb megakadályozza, hogy oda nem illőket is besoroljunk.
    EN: Noise points (-1) are assigned to the nearest topic centre if the cosine similarity is
        at least threshold. HDBSCAN is cautious and leaves many sentences as noise, which is a
        pity for business analysis; the threshold stops misfits from being assigned.
    """
    topics = np.unique(labels[labels >= 0])
    if len(topics) == 0 or (labels >= 0).all():
        return labels.copy()
    centroids = np.stack([embeddings[labels == t].mean(axis=0) for t in topics])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    out = labels.copy()
    noise = np.where(labels < 0)[0]
    sims = embeddings[noise] @ centroids.T
    best = sims.argmax(axis=1)
    ok = sims[np.arange(len(noise)), best] >= threshold
    out[noise[ok]] = topics[best[ok]]
    return out


def _centroids(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    topics = np.unique(labels[labels >= 0])
    cents = np.stack([embeddings[labels == t].mean(axis=0) for t in topics])
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    return topics, cents


def merge_similar_topics(embeddings: np.ndarray, labels: np.ndarray, max_topics: int | None = None,
                         min_similarity: float | None = None,
                         max_topic_share: float | None = None) -> tuple[np.ndarray, list[dict]]:
    """
    HU: TÉMA-ÖSSZEVONÁS (vö. BERTopic "topic reduction"). A HDBSCAN a mondatok MEGFOGALMAZÁSA
        szerint is szétválaszt (pl. "rooms are small" vs. "the room was quite small" — ugyanaz a
        panasz, két fürt). Ezért a finom fürtözés után lépésenként összevonjuk a két leghasonlóbb
        témacsoportot, amíg
          • a témák száma > max_topics, VAGY
          • van két csoport, amelyek hasonlósága ≥ min_similarity (közel-duplikátum).
        Két csoport hasonlósága = a finom témáik középpontjai közötti ÁTLAGOS koszinusz-hasonlóság
        (átlagos kapcsolás). Az összevont csoportnak NEM számolunk új középpontot: az "átlagos"
        középpont minden témához hasonló lenne, és hógolyóként mindent elnyelne (v3-as hiba: a
        zaj aspektusban a forgalom, a zene és a szomszédok egyetlen 55%-os témává olvadt).
        max_topic_share: összevont téma legfeljebb ekkora arányú lehet (biztonsági korlát).
    EN: TOPIC MERGING (cf. BERTopic "topic reduction"). HDBSCAN also separates sentences by their
        WORDING (e.g. "rooms are small" vs. "the room was quite small" — the same complaint, two
        clusters). So after the fine clustering the two most similar topic groups are merged step
        by step while
          • the number of topics is > max_topics, OR
          • two groups have similarity ≥ min_similarity (near-duplicates).
        Similarity of two groups = the AVERAGE cosine similarity between the centres of their fine
        topics (average linkage). No new centre is computed for a merged group: an "average" centre
        would be similar to every topic and would swallow everything like a snowball (the v3 bug:
        in the noise aspect traffic, music and neighbours melted into one 55% topic).
        max_topic_share: a merged topic may not exceed this share (safety cap).

    Returns: new labels (renumbered by size) and the list of merges.
    """
    labels = labels.copy()
    merges: list[dict] = []
    if max_topics is None and min_similarity is None:
        return labels, merges
    topics, cents = (np.unique(labels[labels >= 0]), None)
    if len(topics) < 2:
        return labels, merges
    topics, cents = _centroids(embeddings, labels)
    fine_sim = cents @ cents.T
    sizes = np.array([(labels == t).sum() for t in topics], dtype=float)
    total = float((labels >= 0).sum())
    groups: list[list[int]] = [[i] for i in range(len(topics))]   # HU/EN: indices of fine topics

    while len(groups) > 1:
        best, pair = -np.inf, None
        for a, b in itertools.combinations(range(len(groups)), 2):
            if max_topic_share is not None and \
                    (sizes[groups[a]].sum() + sizes[groups[b]].sum()) / total > max_topic_share:
                continue
            sim = float(fine_sim[np.ix_(groups[a], groups[b])].mean())
            if sim > best:
                best, pair = sim, (a, b)
        if pair is None:
            break                                     # HU/EN: every merge would break the size cap
        too_many = max_topics is not None and len(groups) > max_topics
        too_similar = min_similarity is not None and best >= min_similarity
        if not (too_many or too_similar):
            break
        a, b = pair
        size_a, size_b = sizes[groups[a]].sum(), sizes[groups[b]].sum()
        merges.append({"similarity": round(best, 4), "size_kept": int(max(size_a, size_b)),
                       "size_merged": int(min(size_a, size_b)),
                       "reason": "max_topics" if too_many else "near-duplicate"})
        groups[a] = groups[a] + groups[b]
        del groups[b]

    mapping = {int(topics[i]): g_id for g_id, g in enumerate(groups) for i in g}
    out = np.array([mapping.get(int(x), -1) if x >= 0 else -1 for x in labels])
    return renumber_by_size(out), merges


def assign_to_topics(new_embeddings: np.ndarray, fit_embeddings: np.ndarray, fit_labels: np.ndarray,
                     threshold: float = CFG.topic_outlier_threshold) -> np.ndarray:
    """
    HU: Új mondatok besorolása egy már illesztett témamodellbe: a legközelebbi téma-középpont
        (koszinusz), ha a hasonlóság ≥ threshold, különben -1 (zaj).
    EN: Assigning new sentences to an already fitted topic model: the nearest topic centre
        (cosine) if the similarity is ≥ threshold, otherwise -1 (noise).
    """
    if len(new_embeddings) == 0 or not (fit_labels >= 0).any():
        return np.full(len(new_embeddings), -1)
    topics, cents = _centroids(fit_embeddings, fit_labels)
    sims = new_embeddings @ cents.T
    best = sims.argmax(axis=1)
    return np.where(sims[np.arange(len(best)), best] >= threshold, topics[best], -1)


def subtopic_params() -> dict:
    """HU: Az altéma-modell beállításai a configból. / EN: Subtopic model settings from the config."""
    return {"min_cluster_share": CFG.subtopic_min_cluster_share, "max_topics": CFG.subtopic_max_topics,
            "merge_similarity": CFG.subtopic_merge_similarity, "max_topic_share": CFG.subtopic_max_topic_share}


# ==========================================================================
# 3) LEÍRÁS: c-TF-IDF / DESCRIPTION: c-TF-IDF
# ==========================================================================
def class_tfidf(texts: list[str], labels: np.ndarray, top_n: int = CFG.topic_top_n_terms,
                ngram_range: tuple[int, int] = (1, 2), min_df: int = 3):
    """
    HU: c-TF-IDF (Grootendorst, 2022): a téma összes mondata egy "osztálydokumentum".
            W(t, c) = tf(t, c) / Σ tf(·, c)  ·  log(1 + A / f(t))
        tf(t, c) — a kifejezés gyakorisága a témában, A — átlagos szószám témánként,
        f(t) — a kifejezés gyakorisága az összes témában. A mindenhol gyakori szó
        (pl. "shower" a fürdőszoba-altémákban) így kis súlyt kap.
    EN: c-TF-IDF (Grootendorst, 2022): all sentences of a topic form one "class document".
            W(t, c) = tf(t, c) / Σ tf(·, c)  ·  log(1 + A / f(t))
        tf(t, c) — frequency of the term in the topic, A — average word count per topic,
        f(t) — frequency of the term across all topics. A word frequent everywhere
        (e.g. "shower" in the bathroom subtopics) thus gets a small weight.

    Returns: ({topic: [(term, weight), ...]}, binary doc-term matrix, vocabulary)
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer

    keep = labels >= 0
    min_df = min(min_df, max(1, int(keep.sum() // 50)))
    vectorizer = CountVectorizer(stop_words=stopwords(), ngram_range=ngram_range, min_df=min_df)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:                                            # HU/EN: empty vocabulary
        return {}, None, np.array([])
    vocab = np.array(vectorizer.get_feature_names_out())

    topics = np.unique(labels[keep])
    indicator = sparse.csr_matrix((np.ones(keep.sum()), (np.searchsorted(topics, labels[keep]),
                                   np.where(keep)[0])), shape=(len(topics), len(texts)))
    tf = np.asarray((indicator @ X).todense(), dtype=float)
    freq = tf.sum(axis=0)
    avg_words = tf.sum(axis=1).mean()
    weights = (tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)) * np.log(1 + avg_words / np.maximum(freq, 1))

    terms = {int(t): [(vocab[j], float(weights[i, j])) for j in np.argsort(weights[i])[::-1][:top_n]
                      if weights[i, j] > 0]
             for i, t in enumerate(topics)}
    return terms, (X > 0).astype(np.int32).tocsc(), vocab


def npmi_coherence(doc_term: "sparse.csc_matrix", vocab: np.ndarray, topic_terms: list[str]) -> float:
    """
    HU: NPMI-koherencia (Bouma, 2009; Röder et al., 2015): a téma kulcsszavai mennyire
        fordulnak elő EGYÜTT ugyanabban a mondatban. Tartomány [−1, 1]; 0 felett jó,
        0,1 felett már jól értelmezhető téma szokott lenni.
    EN: NPMI coherence (Bouma, 2009; Röder et al., 2015): how often the topic's keywords
        occur TOGETHER in the same sentence. Range [−1, 1]; above 0 is good, above 0.1
        usually means a well-interpretable topic.
    """
    if doc_term is None:
        return float("nan")
    index = {w: i for i, w in enumerate(vocab)}
    cols = [index[w] for w in topic_terms if w in index]
    if len(cols) < 2:
        return float("nan")
    n_docs = doc_term.shape[0]
    sub = doc_term[:, cols]
    p = np.asarray(sub.sum(axis=0)).ravel() / n_docs
    co = (sub.T @ sub).toarray() / n_docs
    scores = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        if co[i, j] <= 0:
            scores.append(-1.0)
        elif co[i, j] >= 1:
            scores.append(1.0)
        else:
            scores.append(np.log(co[i, j] / (p[i] * p[j])) / -np.log(co[i, j]))
    return float(np.mean(scores))


def topic_diversity(term_lists: list[list[str]]) -> float:
    """
    HU: Témadiverzitás (Dieng et al., 2020): az összes téma kulcsszavai közül hány egyedi.
        1 = a témák teljesen különböznek, alacsony érték = ismétlődő, átfedő témák.
    EN: Topic diversity (Dieng et al., 2020): share of unique words among all topics' keywords.
        1 = topics are fully distinct, a low value = repetitive, overlapping topics.
    """
    words = [w for terms in term_lists for w in terms]
    return len(set(words)) / len(words) if words else float("nan")


# ==========================================================================
# 4) A TELJES TÉMAMODELL / THE FULL TOPIC MODEL
# ==========================================================================
@dataclass
class TopicResult:
    """HU: Egy témamodell eredménye. / EN: Result of one topic model."""
    labels: np.ndarray                         # HU/EN: topic per sentence (-1 = noise)
    topic_info: pd.DataFrame                   # HU/EN: one row per topic
    metrics: dict = field(default_factory=dict)
    merges: list = field(default_factory=list)       # HU/EN: topic merges (merge_similar_topics)


def fit_topics(texts: pd.Series, embeddings: np.ndarray, seed: int = CFG.random_seed,
               min_cluster_size: int | None = None, min_cluster_share: float = CFG.topic_overview_min_share,
               reduce_outliers: bool = True, n_examples: int = 3, max_topics: int | None = None,
               merge_similarity: float | None = None, max_topic_share: float | None = None,
               reduced: np.ndarray | None = None) -> TopicResult:
    """
    HU: Egy témamodell: UMAP -> HDBSCAN -> (zaj áthelyezése) -> c-TF-IDF -> minőség.
        min_cluster_size: alapból max(CFG.topic_min_cluster_size, a mondatok min_cluster_share
        része) — az aspektusonkénti altémáknál CFG.subtopic_min_cluster_share (3%).
    EN: One topic model: UMAP -> HDBSCAN -> (reassign noise) -> c-TF-IDF -> quality.
        min_cluster_size: by default max(CFG.topic_min_cluster_size, min_cluster_share of the
        sentences) — for per-aspect subtopics CFG.subtopic_min_cluster_share (3%).
    max_topics / merge_similarity: optional topic merging (merge_similar_topics).
    reduced: an already computed UMAP output (the granularity sweep reuses it — UMAP is the slow step).
    """
    texts = texts.fillna("").astype(str).reset_index(drop=True)
    n = len(texts)
    min_cluster_size = min_cluster_size or max(CFG.topic_min_cluster_size, int(n * min_cluster_share))

    if reduced is None:
        reduced = reduce_dimensions(embeddings, seed=seed)
    raw_labels = renumber_by_size(cluster(reduced, min_cluster_size))
    noise_before = float((raw_labels < 0).mean())
    labels = reassign_outliers(embeddings, raw_labels) if reduce_outliers else raw_labels
    n_before_merge = int(len(np.unique(labels[labels >= 0])))
    merges: list[dict] = []
    merged_from: dict[int, str] = {}
    if max_topics is not None or merge_similarity is not None:
        fine_terms, _, _ = class_tfidf(texts.tolist(), labels)
        fine_labels = labels
        labels, merges = merge_similar_topics(embeddings, labels, max_topics, merge_similarity, max_topic_share)
        # HU: melyik finom témákból állt össze (átláthatóság) / EN: which fine topics it was built from
        for t in np.unique(labels[labels >= 0]):
            parts = np.unique(fine_labels[labels == t])
            parts = parts[parts >= 0]
            if len(parts) > 1:
                merged_from[int(t)] = " + ".join(" / ".join(w for w, _ in fine_terms[q][:3])
                                                 for q in parts if q in fine_terms)

    terms, doc_term, vocab = class_tfidf(texts.tolist(), labels)
    rows = []
    for t in sorted(terms):
        members = np.where(labels == t)[0]
        centroid = embeddings[members].mean(axis=0)
        closest = members[np.argsort(-(embeddings[members] @ centroid))[:n_examples]]
        words = [w for w, _ in terms[t]]
        rows.append({"topic": t, "size": len(members), "share": len(members) / n,
                     "label": " / ".join(words[:3]), "top_terms": ", ".join(words),
                     "coherence_npmi": npmi_coherence(doc_term, vocab, words),
                     "examples": " || ".join(texts.iloc[closest].str.slice(0, 160)),
                     "merged_from": merged_from.get(int(t), "")})
    info = pd.DataFrame(rows, columns=["topic", "size", "share", "label", "top_terms",
                                       "coherence_npmi", "examples", "merged_from"]).astype(
        {"topic": int, "size": int, "share": float, "coherence_npmi": float})
    metrics = {
        "n_sentences": n, "n_topics": len(info), "n_topics_before_merge": n_before_merge,
        "n_merges": len(merges), "min_cluster_size": min_cluster_size,
        "noise_share_before": round(noise_before, 4), "noise_share_after": round(float((labels < 0).mean()), 4),
        "mean_coherence_npmi": round(float(info["coherence_npmi"].mean()), 4) if len(info) else float("nan"),
        "topic_diversity": round(topic_diversity([t.split(", ") for t in info.get("top_terms", [])]), 4),
    }
    log.info("Topics: %d | noise %.1f%% -> %.1f%% | NPMI %.3f | diversity %.3f", metrics["n_topics"],
             100 * noise_before, 100 * metrics["noise_share_after"], metrics["mean_coherence_npmi"],
             metrics["topic_diversity"])
    return TopicResult(labels=labels, topic_info=info, metrics=metrics, merges=merges)


def topic_stability(texts: pd.Series, embeddings: np.ndarray,
                    seeds: tuple[int, ...] = CFG.topic_stability_seeds, **kwargs) -> pd.DataFrame:
    """
    HU: Stabilitás: ugyanaz a modell különböző véletlen kezdőértékekkel. A párok közötti
        korrigált Rand-index (ARI) azt méri, mennyire ugyanazok a csoportosítások
        (1 = azonos, 0 = véletlenszerű). 0,5 felett a témák érdemben stabilak.
    EN: Stability: the same model with different random seeds. The pairwise adjusted Rand
        index (ARI) measures how similar the groupings are (1 = identical, 0 = random).
        Above 0.5 the topics are substantially stable.
    """
    from sklearn.metrics import adjusted_rand_score

    runs = {seed: fit_topics(texts, embeddings, seed=seed, **kwargs) for seed in seeds}
    rows = [{"seed": seed, **r.metrics} for seed, r in runs.items()]
    aris = [adjusted_rand_score(runs[a].labels, runs[b].labels) for a, b in itertools.combinations(seeds, 2)]
    table = pd.DataFrame(rows).set_index("seed")
    table.attrs["mean_pairwise_ari"] = float(np.mean(aris)) if aris else float("nan")
    log.info("Stability over seeds %s: mean pairwise ARI = %.3f", seeds, table.attrs["mean_pairwise_ari"])
    return table


def granularity_sweep(texts: pd.Series, embeddings: np.ndarray,
                      shares: tuple[float, ...] = CFG.subtopic_sweep_shares,
                      max_topics_grid: tuple[int | None, ...] = CFG.subtopic_sweep_max_topics,
                      merge_similarity: float | None = CFG.subtopic_merge_similarity,
                      max_topic_share: float | None = CFG.subtopic_max_topic_share,
                      seed: int = CFG.random_seed) -> pd.DataFrame:
    """
    HU: Finomság-vizsgálat: ugyanaz a mondathalmaz több fürtméret-aránnyal (min_cluster_share) és
        téma-felső-határral (max_topics; None = nincs összevonás). A lassú UMAP csak EGYSZER fut,
        a fürtözés és az összevonás gyors. Megmutatja a témák számát, a zajt, a koherenciát és a
        legnagyobb téma arányát — ebből indokolható a választott beállítás (dolgozat 3.4).
    EN: Granularity sweep: the same sentences with several minimum cluster shares and topic caps
        (max_topics; None = no merging). The slow UMAP runs only ONCE, clustering and merging are
        fast. It shows the number of topics, noise, coherence and the share of the largest topic —
        this justifies the chosen setting (thesis 3.4).
    """
    reduced = reduce_dimensions(embeddings, seed=seed)
    rows = []
    for share in shares:
        for max_topics in max_topics_grid:
            result = fit_topics(texts, embeddings, seed=seed, min_cluster_share=share, reduced=reduced,
                                max_topics=max_topics,
                                merge_similarity=merge_similarity if max_topics is not None else None,
                                max_topic_share=max_topic_share)
            info = result.topic_info
            rows.append({"min_cluster_share": share, "max_topics": max_topics if max_topics else "none",
                         **{k: result.metrics[k] for k in
                            ("min_cluster_size", "n_topics_before_merge", "n_topics", "noise_share_before",
                             "mean_coherence_npmi", "topic_diversity")},
                         "largest_topic_share": round(float(info["share"].max()), 4) if len(info) else float("nan")})
    return pd.DataFrame(rows).set_index(["min_cluster_share", "max_topics"])


# ==========================================================================
# 5) ALTÉMÁK ASPEKTUSONKÉNT / SUBTOPICS PER ASPECT
# ==========================================================================
def fit_subtopics_per_aspect(sentences: pd.DataFrame, embeddings: np.ndarray, aspects: list[str],
                             text_col: str = "sentence", flag_prefix: str = "asp_",
                             sample_per_aspect: int = CFG.topic_sample_per_aspect,
                             min_sentences: int = CFG.topic_min_sentences,
                             seed: int = CFG.random_seed, **kwargs):
    """
    HU: Minden aspektusra külön témamodell a mondatain (pl. a negatív fürdőszoba-mondatokon).
        sentences és embeddings sorai egymásnak megfelelnek (ugyanaz a sorrend).
    EN: A separate topic model per aspect on its sentences (e.g. the negative bathroom ones).
        Rows of sentences and embeddings correspond (same order).

    Returns
    -------
    topic_table  — aspect | topic | size | share_within_aspect | label | top_terms | NPMI | examples
    assignments  — one row per (sentence, aspect): row (position in `sentences`), aspect, topic, topic_label
    summary      — one row per aspect: n sentences, n topics, noise share, NPMI, diversity
    """
    if len(sentences) != len(embeddings):
        raise ValueError("sentences and embeddings must have the same number of rows")
    rng = np.random.default_rng(seed)
    tables, assigns, summary = [], [], []
    for aspect in aspects:
        if aspect in NON_THEMATIC_ASPECTS:
            continue
        rows = np.where(sentences[f"{flag_prefix}{aspect}"].to_numpy() == 1)[0]
        if len(rows) < min_sentences:
            log.info("%-16s skipped (%d sentences < %d)", aspect, len(rows), min_sentences)
            continue
        all_rows = rows
        if len(rows) > sample_per_aspect:
            rows = np.sort(rng.choice(rows, size=sample_per_aspect, replace=False))
        log.info("%-16s %d sentences (model fitted on %d)", aspect, len(all_rows), len(rows))
        for key, value in subtopic_params().items():
            kwargs.setdefault(key, value)
        result = fit_topics(sentences[text_col].iloc[rows], embeddings[rows], seed=seed, **kwargs)
        summary.append({"aspect": aspect, **result.metrics})
        if result.topic_info.empty:
            log.warning("%-16s no topic found (all sentences are noise) — try a smaller min_cluster_size", aspect)
            continue
        info = result.topic_info.assign(aspect=aspect).rename(columns={"share": "share_within_aspect"})
        tables.append(info)
        label_map = dict(zip(info["topic"], info["label"]))
        # HU: A mintán kívüli mondatok (teljes adatnál) a legközelebbi téma-középponthoz kerülnek,
        #     ugyanazzal a küszöbbel, mint a zajpontok — így a panasztérkép MINDEN mondatot számol.
        # EN: Sentences outside the sample (full data) go to the nearest topic centre, with the same
        #     threshold as noise points — so the complaint map counts EVERY sentence.
        rest = np.setdiff1d(all_rows, rows)
        rest_topics = assign_to_topics(embeddings[rest], embeddings[rows], result.labels)
        both_rows = np.concatenate([rows, rest])
        both_topics = np.concatenate([result.labels, rest_topics])
        assigns.append(pd.DataFrame({"row": both_rows, "aspect": aspect, "topic": both_topics,
                                     "topic_label": [label_map.get(t, "(unassigned)") for t in both_topics],
                                     "in_fit_sample": np.r_[np.ones(len(rows), bool), np.zeros(len(rest), bool)]}))

    cols = ["aspect", "topic", "size", "share_within_aspect", "label", "top_terms", "coherence_npmi", "examples",
            "merged_from"]
    topic_table = pd.concat(tables, ignore_index=True)[cols] if tables else pd.DataFrame(columns=cols)
    assignments = pd.concat(assigns, ignore_index=True) if assigns else pd.DataFrame(
        columns=["row", "aspect", "topic", "topic_label", "in_fit_sample"])
    summary_cols = ["aspect", "n_sentences", "n_topics_before_merge", "n_topics", "min_cluster_size",
                    "noise_share_before",
                    "noise_share_after", "mean_coherence_npmi", "topic_diversity"]
    summary = pd.DataFrame(summary, columns=summary_cols).set_index("aspect")
    if summary.empty:
        log.warning("No aspect reached min_sentences=%d — lower CFG.topic_min_sentences.", min_sentences)
    return topic_table, assignments, summary


def rename_topics(topic_table: pd.DataFrame, assignments: pd.DataFrame,
                  names: dict[tuple[str, int], str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    HU: Kézi téma-elnevezés: names = {("bathroom", 0): "no hot water", ...}. Az automatikus
        címke (a 3 legjellemzőbb kifejezés) kiindulópont; a dolgozatba érdemes beszédes nevet adni.
    EN: Manual topic naming: names = {("bathroom", 0): "no hot water", ...}. The automatic label
        (the 3 most characteristic terms) is a starting point; meaningful names suit the thesis.
    """
    topic_table, assignments = topic_table.copy(), assignments.copy()
    for (aspect, topic), name in names.items():
        topic_table.loc[(topic_table["aspect"] == aspect) & (topic_table["topic"] == topic), "label"] = name
        assignments.loc[(assignments["aspect"] == aspect) & (assignments["topic"] == topic), "topic_label"] = name
    return topic_table, assignments


def segment_table(assignments: pd.DataFrame, sentences: pd.DataFrame, aspect: str, segment_col: str,
                  min_count: int = 50, top_segments: int = 10,
                  top_topics: int | None = CFG.heatmap_max_subtopics, alpha: float = 0.05,
                  min_effect: float = CFG.heatmap_min_effect) -> pd.DataFrame:
    """
    HU: "Panasztérkép": egy aspektus altémáinak aránya szegmensenként (pl. szállástípus).
        A lift = a szegmensbeli arány / az összesített arány; 1 felett az altéma ott gyakoribb.
    EN: "Complaint map": share of an aspect's subtopics per segment (e.g. accommodation type).
        Lift = share in the segment / overall share; above 1 the subtopic is more common there.
        top_topics: only the N largest subtopics are shown (the shares are still computed over
        ALL subtopics, so the lift values do not change — the map just becomes readable).

    HU: SZIGNIFIKANCIA: minden cellára kétmintás arány-z-próba (a szegmens vs. az összes többi),
        Benjamini–Hochberg-korrekcióval (sok cella = sok teszt). lift.attrs: "counts", "q",
        "significant" (q < alpha), "notable" (szignifikáns ÉS |lift − 1| ≥ min_effect). Kis
        szegmensben (n≈80) egy 2-es lift is lehet véletlen; nagy mintán (teljes adat) viszont
        egy 1,06-os, gyakorlatilag jelentéktelen eltérés is szignifikáns — ezért kell mindkét feltétel.
    EN: SIGNIFICANCE: a two-proportion z-test per cell (the segment vs. all others), with
        Benjamini–Hochberg correction (many cells = many tests). lift.attrs: "counts", "q",
        "significant" (q < alpha), "notable" (significant AND |lift − 1| ≥ min_effect). In a small
        segment (n≈80) even a lift of 2 can be chance; on a large sample (full data), however, even a
        practically meaningless 1.06 becomes significant — hence both conditions.
    """
    sub = assignments[(assignments["aspect"] == aspect) & (assignments["topic"] >= 0)].copy()
    sub["segment"] = sentences[segment_col].iloc[sub["row"]].to_numpy()
    sub = sub.dropna(subset=["segment"])
    keep = sub["segment"].value_counts()
    keep = keep[keep >= min_count].nlargest(top_segments).index
    sub = sub[sub["segment"].isin(keep)]
    if sub.empty:
        return pd.DataFrame()
    share = pd.crosstab(sub["segment"], sub["topic_label"], normalize="index")
    overall = sub["topic_label"].value_counts(normalize=True)
    columns = overall.index[:top_topics] if top_topics else overall.index   # HU/EN: largest first
    lift = share[columns] / overall[columns]

    counts = pd.crosstab(sub["segment"], sub["topic_label"])[columns]
    n_seg = sub["segment"].value_counts().reindex(counts.index).to_numpy()[:, None]
    n_all, topic_all = len(sub), sub["topic_label"].value_counts()[columns].to_numpy()[None, :]
    p_value = _two_proportion_p(counts.to_numpy(), n_seg, topic_all - counts.to_numpy(), n_all - n_seg)
    q = benjamini_hochberg(p_value.ravel()).reshape(p_value.shape)
    lift = lift.round(3)
    lift.attrs["n"] = sub["segment"].value_counts().reindex(lift.index).to_dict()
    lift.attrs["counts"] = counts
    lift.attrs["q"] = pd.DataFrame(q, index=counts.index, columns=columns)
    lift.attrs["significant"] = lift.attrs["q"] < alpha
    lift.attrs["notable"] = lift.attrs["significant"] & ((lift - 1).abs() >= min_effect)
    return lift


def _two_proportion_p(x1: np.ndarray, n1: np.ndarray, x2: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """HU: Kétoldali kétmintás arány-z-próba p-értéke. / EN: Two-sided two-proportion z-test p-value."""
    from scipy.stats import norm
    x1, n1, x2, n2 = (np.asarray(v, dtype=float) for v in (x1, n1, x2, n2))
    pooled = (x1 + x2) / np.maximum(n1 + n2, 1)
    se = np.sqrt(np.maximum(pooled * (1 - pooled) * (1 / np.maximum(n1, 1) + 1 / np.maximum(n2, 1)), 1e-12))
    z = (x1 / np.maximum(n1, 1) - x2 / np.maximum(n2, 1)) / se
    return 2 * norm.sf(np.abs(z))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """HU: Benjamini–Hochberg q-értékek (FDR). / EN: Benjamini–Hochberg q-values (FDR)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def segment_long_table(lift: pd.DataFrame) -> pd.DataFrame:
    """
    HU: A panasztérkép hosszú formában (szegmens | altéma | db | lift | q | szignifikáns) — a
        dolgozat szövegéhez: "Japánban a kis szoba panasz 2,45-szörös (n=…, q=…)".
    EN: The complaint map in long format (segment | subtopic | count | lift | q | significant) —
        for the thesis text: "in Japan the small-room complaint is 2.45 times as common (n=…, q=…)".
    """
    long = lift.stack().rename("lift").to_frame()
    long["count"] = lift.attrs["counts"].stack()
    long["segment_n"] = [lift.attrs["n"][seg] for seg in long.index.get_level_values(0)]
    long["q"] = lift.attrs["q"].stack().round(4)
    long["significant"] = lift.attrs["significant"].stack()
    long["notable"] = lift.attrs["notable"].stack()
    long.index.names = ["segment", "subtopic"]
    return long.reset_index().sort_values(["notable", "lift"], ascending=[False, False])


# ==========================================================================
# 6) KÉZI ELLENŐRZÉS / MANUAL CHECK
# ==========================================================================
def export_topic_check(topic_table: pd.DataFrame, assignments: pd.DataFrame, sentences: pd.DataFrame,
                       n_per_topic: int = 20, text_col: str = "sentence", seed: int = CFG.random_seed,
                       filename: str = "topic_check.xlsx") -> Path:
    """
    HU: Témánként n_per_topic véletlen mondat kézi ellenőrzésre. A "fits" oszlopba 1 (illik
        a témához) vagy 0 (nem illik) kerül. Ebből a témák PONTOSSÁGA számolható.
    EN: n_per_topic random sentences per topic for manual checking. Put 1 (fits the topic)
        or 0 (does not fit) into the "fits" column. The topics' PRECISION is computed from it.
    """
    PATHS.ensure()
    path = PATHS.annotation / filename
    if path.exists():
        raise FileExistsError(f"{path} already exists — it may contain your checks.")
    sample = (assignments[assignments["topic"] >= 0]
              .sample(frac=1.0, random_state=seed)
              .groupby(["aspect", "topic"], sort=False).head(n_per_topic)
              .sort_values(["aspect", "topic"]))
    out = sample.assign(sentence=sentences[text_col].iloc[sample["row"]].to_numpy(), fits="")
    out[["aspect", "topic", "topic_label", "sentence", "fits"]].to_excel(path, index=False)
    log.info("Topic check exported: %s (%d sentences)", path, len(out))
    return path


def evaluate_topic_check(filename: str = "topic_check.xlsx") -> pd.DataFrame:
    """HU: Témánkénti pontosság a kézi ellenőrzésből. / EN: Per-topic precision from the manual check."""
    data = pd.read_excel(PATHS.annotation / filename)
    data["fits"] = pd.to_numeric(data["fits"], errors="coerce")
    data = data.dropna(subset=["fits"])
    table = (data.groupby(["aspect", "topic", "topic_label"])["fits"]
                 .agg(checked="size", precision="mean").reset_index())
    log.info("Overall topic precision: %.3f (%d sentences checked)", data["fits"].mean(), len(data))
    return table.round(3)


# ==========================================================================
# 7) ÁBRÁK / FIGURES
# ==========================================================================
def plot_topic_sizes(topic_info: pd.DataFrame, title: str, top_n: int = 15, sentiment: str = "negative",
                     save_as: str | None = None):
    """HU: A legnagyobb témák mérete (a teljes halmaz %-ában). / EN: Size of the largest topics (% of the set)."""
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    share_col = "share" if "share" in topic_info.columns else "share_within_aspect"
    d = topic_info.nlargest(top_n, "size").sort_values("size")
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.4 * len(d) + 1.6)))
    ax.barh(d["label"], d[share_col], color=SENTIMENT_COLORS.get(sentiment, INK["secondary"]), height=0.72)
    for y, v in enumerate(d[share_col]):
        ax.text(v, y, f" {v:.1%}", va="center", fontsize=9, color=INK["secondary"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the sentences")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.set_title(title, loc="left", pad=10)
    plt.tight_layout()
    if save_as:
        save_figure(save_as)
    return ax


def plot_aspect_subtopics(topic_table: pd.DataFrame, aspects: list[str] | None = None, top_k: int = 4,
                          ncols: int = 3, sentiment: str = "negative", save_as: str | None = None):
    """
    HU: Kis többszörösök: aspektusonként egy panel, benne a top_k altéma aránya az aspektuson
        belül. Egy panel = egy adatsor, ezért jelmagyarázat nem kell.
    EN: Small multiples: one panel per aspect with the share of its top_k subtopics within the
        aspect. One panel = one series, so no legend is needed.
    """
    import matplotlib.pyplot as plt
    from .evaluation import save_figure

    if topic_table.empty:
        log.warning("Empty topic table, nothing to plot.")
        return None
    aspects = aspects or (topic_table.groupby("aspect")["size"].sum().sort_values(ascending=False).index.tolist())
    nrows = int(np.ceil(len(aspects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax, aspect in zip(axes.ravel(), aspects):
        d = topic_table[topic_table["aspect"] == aspect].nlargest(top_k, "size").sort_values("size")
        ax.barh(d["label"].str.slice(0, 38), d["share_within_aspect"],
                color=SENTIMENT_COLORS.get(sentiment, INK["secondary"]), height=0.7)
        for y, v in enumerate(d["share_within_aspect"]):
            ax.text(v, y, f" {v:.0%}", va="center", fontsize=8.5, color=INK["secondary"])
        ax.set_title(aspect.replace("_", " "), loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
    for ax in axes.ravel()[len(aspects):]:
        ax.set_visible(False)
    fig.suptitle(f"Largest {sentiment} subtopics within each aspect (share of the aspect's sentences)",
                 x=0.01, ha="left", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_as:
        save_figure(save_as)
    return axes


def plot_segment_heatmap(lift: pd.DataFrame, title: str, save_as: str | None = None):
    """
    HU: Divergens hőtérkép a liftre (kék = ritkább, piros = gyakoribb a szegmensben, mint átlagosan).
    EN: Diverging heatmap of the lift (blue = rarer, red = more common in the segment than on average).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    from .config import DIVERGING_BLUE_RED
    from .evaluation import save_figure

    vals = lift.to_numpy(dtype=float)
    span = max(np.nanmax(np.abs(vals - 1)), 0.15)
    cmap = LinearSegmentedColormap.from_list("div_br", DIVERGING_BLUE_RED)
    norm = TwoSlopeNorm(vmin=max(0.0, 1 - span), vcenter=1.0, vmax=1 + span)
    fig, ax = plt.subplots(figsize=(1.25 * lift.shape[1] + 3.6, 0.55 * lift.shape[0] + 2.4))
    sig = lift.attrs.get("notable", lift.attrs.get("significant"))
    sig = sig.reindex(index=lift.index, columns=lift.columns).to_numpy(dtype=bool) if sig is not None \
        else np.ones_like(vals, dtype=bool)
    # HU: a nem szignifikáns cellák halványak / EN: non-significant cells are faded
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto", alpha=np.where(sig, 1.0, 0.25))
    ax.set_xticks(range(lift.shape[1]), [str(c)[:30] for c in lift.columns], rotation=30, ha="right")
    n = lift.attrs.get("n", {})
    ax.set_yticks(range(lift.shape[0]), [f"{i} (n={n.get(i, '?')})" for i in lift.index])
    ax.grid(visible=False)
    for i in range(lift.shape[0]):
        for j in range(lift.shape[1]):
            v = vals[i, j]
            strong = sig[i, j] and abs(v - 1) > span * 0.55
            ax.text(j, i, f"{v:.2f}" + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=8.5,
                    color="white" if strong else (INK["primary"] if sig[i, j] else INK["muted"]))
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Share in segment / overall share", fontsize=9.5)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", pad=12)
    fig.text(0.01, 0.005, f"* significant (two-proportion z-test, Benjamini–Hochberg q < 0.05) and "
             f"|lift − 1| ≥ {CFG.heatmap_min_effect:g}; faded = not", fontsize=8.5, color=INK["muted"],
             ha="left", va="bottom")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    if save_as:
        save_figure(save_as)
    return ax
