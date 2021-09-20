import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Set, Optional, List


from code.ml.models import LSTMHypothesesClassifier
from code.ml.vocab import vocab
from scorer_abstract import (
    ReducedCompoundAnalysis, HypothesesSample, ScorerAbstract
)


class LSTMScorer(ScorerAbstract):
    def __init__(
            self,
            n_vocab: int,
            n_rules: int,
            d_input: int,
            d_rnn: int,
            n_layers: int = 2,
            emb_dropout: float = 0.0,
            rnn_dropout: float = 0.0,
            pretrained_path: Optional[str] = None
    ):
        super().__init__()

        self.model = LSTMHypothesesClassifier(
            n_vocab=n_vocab,
            n_rules=n_rules,
            d_input=d_input,
            d_rnn=d_rnn,
            n_layers=n_layers,
            emb_dropout=emb_dropout,
            rnn_dropout=rnn_dropout
        )
        if pretrained_path:
            self.model.load_state_dict(
                torch.load(
                    pretrained_path, map_location=torch.device('cpu')
                ), strict=False
            )

        self.loss_fn = nn.BCELoss(reduction="sum")
        self.rule2id = dict()

    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        batch = self.prepare_model_input(
            sample.word_c, sample.pos_c, sample.analyses
        )
        res = self.model(**batch)
        return res.squeeze().argmax(-1)

    def prepare_model_input(
            self, word_c, pos_c, analyses: List[ReducedCompoundAnalysis]
    ):
        chars_c = vocab.tokenize(word_c)
        chars_ms, chars_hs = [], []
        rule_ids_c, rule_ids_m, rule_ids_h = [], [], []
        padding_mask_ms, padding_mask_hs = [], []

        max_len_m, max_len_h = 0, 0
        for i in range(len(analyses)):
            a = analyses[i]
            rule_c, rule_l, rule_r = a.rule_c, a.rule_l, a.rule_r
            chars_word_m = vocab.tokenize(a.word_l)
            chars_word_h = vocab.tokenize(a.word_r)
            max_len_m = max(max_len_m, len(chars_word_m))
            max_len_h = max(max_len_h, len(chars_word_h))
            chars_ms.append(chars_word_m)
            chars_hs.append(chars_word_h)
            if rule_c not in self.rule2id:
                self.rule2id[rule_c] = len(self.rule2id)
            if rule_l not in self.rule2id:
                self.rule2id[rule_l] = len(self.rule2id)
            if rule_r not in self.rule2id:
                self.rule2id[rule_r] = len(self.rule2id)
            rule_ids_c.append(self.rule2id[rule_c])
            rule_ids_m.append(self.rule2id[rule_l])
            rule_ids_h.append(self.rule2id[rule_r])

        for i in range(len(chars_ms)):
            l_m = len(chars_ms[i])
            l_h = len(chars_hs[i])
            chars_ms[i].extend([vocab.pad_idx] * (max_len_m - l_m))
            chars_hs[i].extend([vocab.pad_idx] * (max_len_h - l_h))
            padding_mask_ms.append([1] * l_m + [0] * (max_len_m - l_m))
            padding_mask_hs.append([1] * l_h + [0] * (max_len_h - l_h))

        return {
            "chars_c": torch.tensor(chars_c, dtype=torch.int64),
            "chars_ms": torch.tensor(chars_ms, dtype=torch.int64),
            "chars_hs": torch.tensor(chars_hs, dtype=torch.int64),

            "rule_ids_c": torch.tensor(rule_ids_c, dtype=torch.int64),
            "rule_ids_m": torch.tensor(rule_ids_m, dtype=torch.int64),
            "rule_ids_h": torch.tensor(rule_ids_h, dtype=torch.int64),

            "padding_mask_ms": torch.tensor(padding_mask_ms, dtype=torch.bool),
            "padding_mask_hs": torch.tensor(padding_mask_hs, dtype=torch.bool),
        }


scorer = LSTMScorer(
    len(vocab.i2w), 250, 128, 256, emb_dropout=0.1, rnn_dropout=0.25,
    pretrained_path="../../data/models/is_compound_clf_lstm_30_best.bin"
)
#
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
#
# scorer.fit([sample], [2])

# compute_metrics(
#     scorer,
#     "../../data/hypotheses/generated.txt",
#     "../../data/gold_analyses/test.csv"
# )
