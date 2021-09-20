import pandas as pd
from typing import List, Dict, Tuple

from scorer_abstract import ReducedCompoundAnalysis


def matched(
        lhs: ReducedCompoundAnalysis,
        rhs: ReducedCompoundAnalysis
) -> bool:
    return all(
        [
            # lhs.rule_c == rhs.rule_c,
            lhs.word_l == rhs.word_l,
            # lhs.pos_l == rhs.pos_l,
            lhs.word_r == rhs.word_r,
            # lhs.pos_r == rhs.pos_r,
        ]
    )


def read_gold(
        path: str
) -> Dict[Tuple[str, str], ReducedCompoundAnalysis]:
    df = pd.read_csv(path)
    gold_dict = dict()
    for i in range(len(df)):
        # rule,word_d,pos_d,words_m,poss_m,word_h,pos_h
        r = df.iloc[i]
        a_gold = ReducedCompoundAnalysis(
            r["rule"],
            r["words_m"], r["poss_m"], "-",
            r["word_h"], r["pos_h"], "-"
        )
        gold_dict[(r["word_d"], r["pos_d"])] = a_gold
    return gold_dict


def read_analyses(
        path: str
) -> Dict[Tuple[str, str], List[ReducedCompoundAnalysis]]:
    analyses_dict = dict()

    with open(path, 'r') as f:
        for line in f:
            (
                word_c, pos_c, rule_c,
                word_l, pos_l, rule_l,
                word_r, pos_r, rule_r
            ) = line.strip().split('\t')

            a = ReducedCompoundAnalysis(
                rule_c,
                word_l, pos_l, rule_l,
                word_r, pos_r, rule_r
            )
            key = (word_c, pos_c)
            if key not in analyses_dict:
                analyses_dict[key] = []
            analyses_dict[key].append(a)
    return analyses_dict

# data = read_analyses("../../data/hypotheses/generated.txt")
# print(data)
