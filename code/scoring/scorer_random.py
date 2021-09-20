import numpy as np
from typing import Optional

from scorer_abstract import HypothesesSample, ScorerAbstract


class RandomScorer(ScorerAbstract):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        best_idx: int = -1
        best_score = -1
        for i in range(len(sample)):
            score = np.random.random()
            if score > best_score:
                best_idx = i
                best_score = score
        return best_idx
