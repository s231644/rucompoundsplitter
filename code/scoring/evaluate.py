from typing import Optional
from tqdm import tqdm

from scorer_abstract import (
    HypothesesSample, ReducedCompoundAnalysis, ScorerAbstract
)
from utils import read_analyses, read_gold, matched


def evaluate_fsm_generation(
        preds_path: str,
        golds_path: str
):
    preds = read_analyses(preds_path)
    golds = read_gold(golds_path)

    tp, fp, fn = 0, 0, 0
    for k in tqdm(golds):
        if k not in preds:
            print(k, "FN", golds[k])
            fn += 1
            continue

        word_c, pos_c = k
        a_gold = golds[k]
        a_preds = preds[k]

        correct_ids = [
            i for i, a_pred in enumerate(a_preds) if matched(a_gold, a_pred)
        ]
        # TODO: make sure that always len(correct_ids) <= 1.
        if not correct_ids:
            print(k, "FN", golds[k], "FP", a_preds)
            fn += 1
            fp += len(a_preds)
        else:
            tp += 1
            fp += len(a_preds) - 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(
        f"tp={tp} fp={fp} fn={fn} "
        f"P={precision} R={recall} F1={f1}"
    )


def compute_metrics(
        model: ScorerAbstract,
        preds_path: str,
        golds_path: str
):
    preds = read_analyses(preds_path)
    golds = read_gold(golds_path)

    tp, fp, fn = 0, 0, 0
    for k in tqdm(golds):
        if k not in preds:
            fn += 1
            continue

        word_c, pos_c = k
        a_gold = golds[k]
        a_preds = preds[k]
        h = HypothesesSample(word_c, pos_c, a_preds)

        best_idx = model._predict_sample(h)
        if best_idx == -1:
            fn += 1
            continue

        correct_ids = [
            i for i, a_pred in enumerate(a_preds) if matched(a_gold, a_pred)
        ]
        # TODO: make sure that always len(correct_ids) <= 1.
        if best_idx in correct_ids:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = tp / len(golds)
    print(
        f"A={accuracy} "
        # f"tp={tp} fp={fp} fn={fn} "
        # f"P={precision} R={recall} F1={f1}"
    )
    return accuracy


def compare(
        model: ScorerAbstract,
        preds_path: str,
        golds_path: str
):
    preds = read_analyses(preds_path)
    golds = read_gold(golds_path)

    tp = 0
    for k in tqdm(golds):
        if k not in preds:
            print(f"Word {k} not fouds in the generated hypotheses!")
            continue

        word_c, pos_c = k
        a_gold = golds[k]
        a_preds = preds[k]
        h = HypothesesSample(word_c, pos_c, a_preds)

        best_idx = model._predict_sample(h)
        if best_idx == -1:
            continue

        correct_ids = [
            i for i, a_pred in enumerate(a_preds) if matched(a_gold, a_pred)
        ]
        # TODO: make sure that always len(correct_ids) <= 1.
        if best_idx in correct_ids:
            tp += 1
            print(f"+ {k} {a_preds[best_idx]}")
        else:
            print(f"- {k} {a_preds[best_idx]}; correct:\n {a_gold}")
    accuracy = tp / len(golds)
    print(f"A={accuracy}")
    return accuracy


evaluate_fsm_generation(
    "../../data/hypotheses/generated.txt",
    "../../data/gold_analyses/test.csv"
)

# from scorer_freqs import PMIFrequencyScorer, UnigramFrequencyScorer
# from scorer_word2vec import Word2VecScorer
# from scorer_random import RandomScorer
#
# import numpy as np
#
# accs = []
# for i in range(100):
#     model = RandomScorer(seed=i)
#     a = compute_metrics(
#         model,
#         "../../data/hypotheses/generated.txt",
#         "../../data/gold_analyses/test.csv"
#     )
#     accs.append(a)
# print(np.mean(accs), np.std(accs))


# model = UnigramFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     aggregation="mult"
# )


# model = UnigramFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     do_inflection=False,
#     do_capitalization=True,
#     do_uppercase=True
# )

# model = UnigramFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     do_inflection=False,
#     do_capitalization=True,
#     do_uppercase=True
# )

# model = PMIFrequencyScorer(
#     "../../data/ngram_freqs/1grams-3.txt",
#     "../../data/ngram_freqs/2grams-3.txt",
#     do_inflection=True,
#     # do_capitalization=True,
#     # do_uppercase=True
# )

# model = Word2VecScorer(
#     "../../data/embeddings/model.bin", cordiero=False
# )
#
# compute_metrics(
#     model,
#     "../../data/hypotheses/generated.txt",
#     "../../data/gold_analyses/test.csv"
# )
