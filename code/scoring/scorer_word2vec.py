import gensim
import numpy as np
from numpy import dot
from numpy.linalg import norm
from typing import Optional, List

from scorer_abstract import (
    ReducedCompoundAnalysis, HypothesesSample, ScorerAbstract
)


def enorm(x, eps=1e-6):
    return norm(x) + eps


class Word2VecScorer(ScorerAbstract):
    def __init__(
            self,
            embeddings_path: str,
            cordiero: bool = False
    ):
        super().__init__()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_path, binary=True
        )
        self.word_to_pos = {}
        for k in self.model.key_to_index:
            word, pos = k.split('_')
            if word not in self.word_to_pos:
                self.word_to_pos[word] = []
            self.word_to_pos[word].append(pos)

        self.cordiero = cordiero

    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        word_c = sample.word_c
        pos_c = sample.pos_c
        vec_c = self.get_embedding(word_c, pos_c)
        best_idx: int = -1
        best_score = -2
        for i in range(len(sample)):
            analysis = sample[i]
            word_l = analysis.word_l
            pos_l = analysis.pos_l
            word_r = analysis.word_r
            pos_r = analysis.pos_r
            vec_l = self.get_embedding(word_l, pos_l)
            vec_r = self.get_embedding(word_r, pos_r)

            if self.cordiero:
                vec_lr = vec_l / (enorm(vec_l) + vec_r / enorm(vec_r))
                score = dot(vec_c, vec_lr) / (enorm(vec_c) * enorm(vec_lr))
            else:
                score = dot(vec_l, vec_r) / (enorm(vec_l) * enorm(vec_r))
            if score > best_score:
                best_idx = i
                best_score = score
        return best_idx

    def get_embedding(
            self, word: str, pos: Optional[str] = None
    ) -> np.array:
        token = self.get_vocab_token(word, pos)
        if not token:
            vec = np.zeros(self.model.vector_size)
        else:
            vec = np.array(self.model[token])
        return vec

    def get_vocab_token(
            self, word: str, pos: Optional[str] = None
    ) -> Optional[str]:
        w = word.lower()
        poss = self.word_to_pos.get(w, [])
        if not poss:
            if 'ё' in w:
                return self.get_vocab_token(w.replace('ё', 'е'), pos)
            return None
        if pos:
            if pos == 'noun' and word[0].isupper():
                pos_end = 'PROPN'
            else:
                pos_end = pos.upper()
            if pos_end not in poss:
                pos_end = poss[0]
        else:
            pos_end = poss[0]
        return "_".join([w, pos_end])


# a1 = ReducedCompoundAnalysis(
#     "xxx", "Земля", "noun", "itfx", "мера", "noun", "sfx"
# )
# a2 = ReducedCompoundAnalysis(
#     "xxx", "Земля", "noun", "itfx", "мерить", "verb", "sfx"
# )
# a3 = ReducedCompoundAnalysis(
#     "xxx", "земля", "noun", "itfx", "мерить", "verb", "sfx"
# )
#
# sample = HypothesesSample(
#     "землемерие", "noun", [a1, a2, a3]
# )


# b1 = ReducedCompoundAnalysis(
#     "xxx", "запад", "noun", "itfx", "Сибирь", "noun", "sfx"
# )
# b2 = ReducedCompoundAnalysis(
#     "xxx", "западный", "adj", "itfx", "Сибирь", "noun", "sfx"
# )
# b3 = ReducedCompoundAnalysis(
#     "xxx", "запад", "noun", "itfx", "сибирский", "adj", "sfx"
# )
# b4 = ReducedCompoundAnalysis(
#     "xxx", "западный", "adj", "itfx", "сибирский", "adj", "sfx"
# )
# b5 = ReducedCompoundAnalysis(
#     "xxx", "западно", "adv", "itfx", "сибирский", "adj", "sfx"
# )
#
#
# sample = HypothesesSample(
#     "западносибирский", "adj", [b1, b2, b3, b4, b5]
# )
#
# c = Word2VecScorer(
#     "../../data/embeddings/model.bin", cordiero=True
# )
#
# z = c.predict_sample(sample)
# print(z)
