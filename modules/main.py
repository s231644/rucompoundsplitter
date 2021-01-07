import numpy as np
import dynet as dy

from modules.compound_split_scorer import CompoundSplitScorer, CompoundSplitClf
from modules.utils import train, validate, predict, ALPHABET, POSS, RULES


# net = CompoundSplitScorer(
#     n_vocab=len(ALPHABET),
#     n_rules=len(RULES), n_pos=len(POSS),
#     d_input=32, d_rule=32, d_pos=32,
#     d_hidden=32, n_layers=1, attn_type="att"
# )

net = CompoundSplitClf(
    n_vocab=len(ALPHABET),
    n_rules=len(RULES), n_pos=len(POSS),
    d_input=16, d_rule=16, d_pos=16,
    d_hidden=16, n_layers=1, attn_type="att"
)


trainer = dy.AdamTrainer(net.model)

orig_data = []

# for line in json.load(open("../data/data_combined.json")):
#     c, (pos, negs) = line
#     orig_data.append((c, [pos] + negs))

for line in open("../data/data_adj.txt"):
    word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h, y_true = line.strip().split('\t')
    orig_data.append(
        (
            word_c.lower(), pos_c, rule_c,
            word_m.lower(), pos_m, rule_m,
            word_h.lower(), pos_h, rule_h,
            int(y_true)
        )
    )

np.random.shuffle(orig_data)

TRAIN_SIZE = int(0.8 * len(orig_data))
train_data, val_data = orig_data[:TRAIN_SIZE], orig_data[TRAIN_SIZE:]

for i in range(50):
    train_loss, train_acc = train(net, trainer, train_data)
    val_loss, val_acc = validate(net, val_data)
    print(train_loss, train_acc, val_loss, val_acc)
    if train_loss == 0:
        break

val_predict = predict(net, val_data)
for x in val_predict:
    print(x)
