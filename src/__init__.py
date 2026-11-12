"""Text-based sentiment labelling and analysis of hotel reviews with BERT.

The notebooks import the modules after adding the project root to the path:

    import sys; sys.path.insert(0, "..")
    from src.config import CFG, PATHS
"""

__version__ = "0.2.0"

import logging

# Uniform logging in every module (which step is running, how far along).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def set_seed(seed: int | None = None) -> int:
    """
    Fix every random number generator with one call (in the first notebook cell).
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
