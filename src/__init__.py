"""
Szakdolgozat: Sentiment Analysis of Customer Reviews using Machine Learning Techniques

A forrásmodulok belépési pontja. Notebookból így használd:

    import sys; sys.path.append("..")          # a notebooks/ mappából
    from src.config import CFG, PATHS
    from src import data_loading, preprocessing, models_classical, evaluation, aspects
"""

__version__ = "0.1.0"

import logging

# Egységes naplózás minden modulban. A notebookokban így látod majd,
# melyik lépés hol tart — nagy adaton ez nagyon hasznos.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def set_seed(seed: int | None = None) -> int:
    """
    Minden véletlenszám-generátor rögzítése egyetlen hívással.

    Ezt a notebook ELSŐ cellájában hívd meg. Enélkül minden futtatásnál
    kicsit más eredményt kapsz, és a dolgozatba írt számokat nem tudod
    reprodukálni — ami a védésen kellemetlen kérdéseket szül.
    """
    import os
    import random
    from .config import CFG

    seed = CFG.random_seed if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
