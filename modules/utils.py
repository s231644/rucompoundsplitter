from tqdm import tqdm
import dynet as dy
import numpy as np
from typing import Optional, List

DYNET_ZERO = dy.constant(1, 0)


ALPHABET = "<> -ёйцукенгшщзхъфывапролджэячсмитьбю"
l2idx = {l: i for i, l in enumerate(ALPHABET)}

POSS = [
    'UNK',
    'adj',
    'adj_SHORT',
    'adv',
    'comp',
    'glroot',
    'noun',
    'noun_ACC',
    'noun_GEN',
    'num',
    'part',
    'prefixoid',
    'pron',
    'prop',
    'suffixoid',
    'tgr',
    'verb',
    'verb_IMPER',
    'comporative'
]
pos2idx = {p: i for i, p in enumerate(POSS)}

RULES = [
    '<UNK>',
    'rule550([adj + ITFX] + noun -> noun)',
    'rule550([noun + ITFX] + noun -> noun)',
    'rule550([num + ITFX] + noun -> noun)',
    'rule552([пол] + noun + GEN -> noun)',
    'rule570([adj + ITFX] + noun + ник/атник/арник/овник -> noun)',
    'rule570([noun + ITFX] + noun + ник/атник/арник/овник -> noun)',
    'rule570([num + ITFX] + noun + ник/атник/арник/овник -> noun)',
    'rule571([adj + ITFX] + noun + щик/чик/овщик -> noun)',
    'rule572([adj + ITFX] + noun + к(а) -> noun)',
    'rule572([noun + ITFX] + noun + к(а) -> noun)',
    'rule572([num + ITFX] + noun + к(а) -> noun)',
    'rule573([adj + ITFX] + noun + j(е) -> noun)',
    'rule573([num + ITFX] + noun + j(е) -> noun)',
    'rule576([adj + ITFX] + noun + ан-ин/чан-ин -> noun)',
    'rule580([adj + ITFX] + noun + 0 -> noun)',
    'rule580([noun + ITFX] + noun + 0 -> noun)',
    'rule754([adj + ITFX] + adj -> adj)',
    'rule754([adj + ITFX] + part -> adj)',
    'rule754([noun + ITFX] + adj -> adj)',
    'rule754([noun + ITFX] + part -> adj)',
    'rule754([num + ITFX] + adj -> adj)',
    'rule776([adv] + adj -> adj)',
    'rule776([adv] + part -> adj)',
    'rule961([noun + ITFX] + verb -> verb)',
    'rule961([полу/само] + verb -> verb)',
    'rule963([noun] + verb -> verb)'
]
rule2id = {r: i for i, r in enumerate(RULES)}


def ls2is(l):
    return list(map(lambda x: l2idx[x], f"<{l}>"))


def compute_hinge_loss(
        scores: List[dy.Expression],
        correct_id: Optional[int] = 0,
        min_margin: Optional[float] = 0
) -> dy.Expression:
    elem_loss = []
    for i in range(len(scores)):
        margin_i = scores[i] - scores[correct_id] + min_margin
        if margin_i.value() <= 0:
            margin_i *= 0
        elem_loss.append(margin_i)
    if not elem_loss:
        return dy.Expression(DYNET_ZERO)
    else:
        return dy.esum(elem_loss)


def train(model, trainer, data):
    loss_epoch = []
    correct_count = 0
    for word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h, y_true in tqdm(data):
        dy.renew_cg()
        score = model.forward(
            ls2is(word_c), pos2idx[pos_c],
            ls2is(word_m), pos2idx[pos_m],
            ls2is(word_h), pos2idx[pos_h],
            rule2id.get(rule_c, 0)
        )
        if y_true == 1:
            loss = -dy.log(dy.logistic(score))
        else:
            loss = -dy.log(1 - dy.logistic(score))
        loss_epoch.append(loss.value())
        loss.backward()
        trainer.update()
        if bool(score.value() >= 0) == y_true:
            correct_count += 1


    # for (tokens_c, pos_c), analyses in tqdm(data):
    #     dy.renew_cg()
    #     scores = [
    #         model.forward(
    #             ls2is(tokens_c), pos2idx[pos_c],
    #             ls2is(tokens_m), pos2idx[pos_m],
    #             ls2is(tokens_h), pos2idx[pos_h],
    #             rule2id.get(rule_id, 0)
    #         )
    #         for (tokens_m, pos_m), (tokens_h, pos_h), rule_id in analyses
    #     ]
    #     loss = compute_hinge_loss(scores, min_margin=1)
    #     loss_epoch.append(loss.value())
    #     loss.backward()
    #     trainer.update()
    #     scores_np = np.array(list(map(lambda x: x.value(), scores)))
    #     if scores_np.argmax() == 0:
    #         correct_count += 1
    acc = correct_count / len(data)
    return np.mean(loss_epoch), acc


def validate(model, data):
    loss_epoch = []
    correct_count = 0

    for word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h, y_true in tqdm(data):
        dy.renew_cg()
        score = model.forward(
            ls2is(word_c), pos2idx[pos_c],
            ls2is(word_m), pos2idx[pos_m],
            ls2is(word_h), pos2idx[pos_h],
            rule2id.get(rule_c, 0)
        )
        if y_true == 1:
            loss = -dy.log(dy.logistic(score))
        else:
            loss = -dy.log(1 - dy.logistic(score))
        loss_epoch.append(loss.value())

        if bool(score.value() >= 0) == y_true:
            correct_count += 1


    # for (tokens_c, pos_c), analyses in tqdm(data):
    #     dy.renew_cg()
    #     scores = [
    #         model.forward(
    #             ls2is(tokens_c), pos2idx[pos_c],
    #             ls2is(tokens_m), pos2idx[pos_m],
    #             ls2is(tokens_h), pos2idx[pos_h],
    #             rule2id.get(rule_id, 0)
    #         )
    #         for (tokens_m, pos_m), (tokens_h, pos_h), rule_id in analyses
    #     ]
    #     loss = compute_hinge_loss(scores, min_margin=1)
    #     loss_epoch.append(loss.value())
    #     scores_np = np.array(list(map(lambda x: x.value(), scores)))
    #     if scores_np.argmax() == 0:
    #         correct_count += 1
    acc = correct_count / len(data)
    return np.mean(loss_epoch), acc


def predict(model, data):
    predictions = []
    for word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h, y in tqdm(data):
        dy.renew_cg()
        score = model.forward(
            ls2is(word_c), pos2idx[pos_c],
            ls2is(word_m), pos2idx[pos_m],
            ls2is(word_h), pos2idx[pos_h],
            rule2id.get(rule_c, 0)
        )
        predictions.append((word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h, y, score.value()))

    # for (tokens_c, pos_c), analyses in tqdm(data):
    #     dy.renew_cg()
    #     scores = [
    #         model.forward(
    #             ls2is(tokens_c), pos2idx[pos_c],
    #             ls2is(tokens_m), pos2idx[pos_m],
    #             ls2is(tokens_h), pos2idx[pos_h],
    #             rule2id.get(rule_id, 0)
    #         )
    #         for (tokens_m, pos_m), (tokens_h, pos_h), rule_id in analyses
    #     ]
    #     scores_np = np.array(list(map(lambda x: x.value(), scores)))
    #     predictions.append(((tokens_c, pos_c), analyses[scores_np.argmax()]))
    return predictions
