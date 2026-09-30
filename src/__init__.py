"""
Thesis: Sentiment Analysis of Customer Reviews using Machine Learning Techniques

Entry point of the source modules. Usage from a notebook:

    import sys; sys.path.append("..")          # from the notebooks/ folder
    from src.config import CFG, PATHS
    from src import data_loading, preprocessing, models_classical, evaluation, aspects
"""

__version__ = "0.1.0"

import logging

# Uniform logging in every module. In the notebooks this shows which step
# is running and how far along it is — very useful on large data.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def set_seed(seed: int | None = None) -> int:
    """
    Fix every random number generator with a single call.

    Call this in the FIRST cell of the notebook. Without it every run gives
    slightly different results, and the numbers written into the thesis
    cannot be reproduced — which leads to awkward questions at the defence.
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
