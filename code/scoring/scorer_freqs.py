import numpy as np
from tqdm import tqdm
from typing import Dict, Set, List

from scorer_abstract import (
    ReducedCompoundAnalysis, HypothesesSample, ScorerAbstract
)


class UnigramFrequencyScorer(ScorerAbstract):
    def __init__(
            self,
            unigram_counts_path: str,
            aggregation: str = "add",
            do_inflection: bool = False,
            do_capitalization: bool = False,
            do_uppercase: bool = False,
            do_lowercase: bool = False,
            do_yo_conversion: bool = False
    ):
        super().__init__()

        self.unigram_counts = dict()
        with open(unigram_counts_path, 'r') as f:
            for line in tqdm(f, desc="Reading unigram counts"):
                # 6829968	и
                count, form = line.strip().split()
                self.unigram_counts[form] = int(count)

        self.aggregation = aggregation

        self.do_inflection = do_inflection
        self.do_capitalization = do_capitalization
        self.do_uppercase = do_uppercase
        self.do_lowercase = do_lowercase
        self.do_yo_conversion = do_yo_conversion

        if do_inflection:
            import pymorphy2
            self.morph = pymorphy2.MorphAnalyzer()

    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        best_idx: int = -1
        best_score = -1
        for i in range(len(sample)):
            analysis = sample[i]
            word_l = analysis.word_l
            word_r = analysis.word_r
            count_l = self.get_count(word_l)
            count_r = self.get_count(word_r)
            if self.aggregation == "add":
                score = count_l + count_r
            elif self.aggregation == "mult":
                score = max(1, count_l) * max(1, count_r)
            else:
                raise ValueError
            if score > best_score:
                best_idx = i
                best_score = score
        return best_idx

    def get_all_forms(self, x: str) -> Set[str]:
        forms = {x}
        if self.do_inflection:
            morph_parses = self.morph.parse(x)
            for p in morph_parses:
                for word_form in p.lexeme:
                    forms.add(word_form.word)
        cased_forms = set()
        if self.do_capitalization:
            cased_forms |= {f.capitalize() for f in forms}
        if self.do_uppercase:
            cased_forms |= {f.upper() for f in forms}
        if self.do_lowercase:
            cased_forms |= {f.lower() for f in forms}
        forms |= cased_forms
        if self.do_yo_conversion:
            forms |= {f.replace("ё", "е").replace("Ё", "Е") for f in forms}

        return forms

    def get_count_dict(self, x: str, default_value=0) -> Dict[str, int]:
        forms = self.get_all_forms(x)
        return {f: self.unigram_counts.get(f, default_value) for f in forms}

    def get_count(self, x: str, default_value=0) -> int:
        count_dict = self.get_count_dict(x, default_value=default_value)
        c = sum(count_dict.values())
        return c


class PMIFrequencyScorer(UnigramFrequencyScorer):
    def __init__(
            self,
            unigram_counts_path: str,
            bigram_counts_path: str,
            *args, **kwargs
    ):
        super().__init__(
            unigram_counts_path, *args, aggregation="pmi", **kwargs
        )

        self.bigram_counts = dict()
        with open(bigram_counts_path, 'r') as f:
            for line in tqdm(f, desc="Reading bigram counts"):
                #  178869	и		не
                tokens = line.strip().split()
                count, form_l, form_r = tokens[0], tokens[1], tokens[-1]
                self.bigram_counts[(form_l, form_r)] = int(count)

    def _predict_sample(
            self, sample: HypothesesSample
    ) -> int:
        best_idx: int = -1
        best_score = -np.inf
        for i in range(len(sample)):
            analysis = sample[i]
            word_l = analysis.word_l
            word_r = analysis.word_r

            count_l = self.get_count(word_l)
            count_r = self.get_count(word_r)

            # TODO: get rid of a double get_all_forms computation
            forms_l = self.get_all_forms(word_l)
            forms_r = self.get_all_forms(word_r)

            bigrams = set()
            for f_l in forms_l:
                for f_r in forms_r:
                    bigrams.add((f_l, f_r))
                    bigrams.add((f_r, f_l))
            count_b = sum(self.bigram_counts.get(b, 0) for b in bigrams)
            num = max(1, count_b)
            den = max(1, count_l) * max(1, count_r)
            score = np.log2(num / den)
            if score > best_score:
                best_idx = i
                best_score = score
        return best_idx


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
# c = UnigramFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     do_inflection=False,
#     do_capitalization=True
# )

# c = PMIFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     "../../data/ngram_freqs/2grams-3.txt",
#     do_inflection=True,
#     do_capitalization=True,
#     do_uppercase=True
# )

# z = c.predict_sample(sample)
# print(z)
