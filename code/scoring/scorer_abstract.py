from abc import ABC, abstractmethod
from typing import List


class ReducedCompoundAnalysis:
    def __init__(
            self,
            rule_c: str,
            word_l: str, pos_l: str, rule_l: str,
            word_r: str, pos_r: str, rule_r: str,
    ):
        """
        rule567([noun + ITFX] + verb + ниj(е) -> noun)
        Земля
        noun
        ruleINTERFIX(noun)
        мерить
        verb
        rule256(verb + ниj(е) -> noun)
        """
        self.rule_c = rule_c
        self.word_l, self.pos_l, self.rule_l = word_l, pos_l, rule_l
        self.word_r, self.pos_r, self.rule_r = word_r, pos_r, rule_r

    def __str__(self):
        return f"{self.rule_c}|{self.word_l}|{self.word_r}"

    def __repr__(self):
        return f"{self.rule_c}|{self.word_l}|{self.word_r}"


class CompoundAnalysis(ReducedCompoundAnalysis):
    def __init__(
            self,
            word_c: str, pos_c: str, rule_c: str,
            word_l: str, pos_l: str, rule_l: str,
            word_r: str, pos_r: str, rule_r: str,
    ):
        """
        землемерие
        noun
        rule567([noun + ITFX] + verb + ниj(е) -> noun)
        Земля
        noun
        ruleINTERFIX(noun)
        мерить
        verb
        rule256(verb + ниj(е) -> noun)
        """
        super(CompoundAnalysis, self).__init__(
            rule_c,
            word_l, pos_l, rule_l,
            word_r, pos_r, rule_r,
        )
        self.word_c, self.pos_c = word_c, pos_c


class HypothesesSample:
    def __init__(
            self,
            word_c: str, pos_c: str,
            analyses: List[ReducedCompoundAnalysis]
    ):
        self.word_c, self.pos_c = word_c, pos_c
        self.analyses = analyses

    def __getitem__(self, item):
        return self.analyses[item]

    def __len__(self):
        return len(self.analyses)


class ScorerAbstract(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        raise NotImplementedError

    def predict_sample(
            self, sample: HypothesesSample
    ) -> ReducedCompoundAnalysis:
        best_idx = self._predict_sample(sample)
        if best_idx == -1:
            raise IndexError("Nothing to score!")
        return sample[best_idx]
