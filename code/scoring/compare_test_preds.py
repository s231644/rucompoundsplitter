import torch

from code.ml.vocab import vocab
from evaluate import compare
from train_lstm_scorer import LSTMScorer


best_path = "scorer_best.bin"

scorer = LSTMScorer(
    len(vocab.i2w), 250, 128, 256, emb_dropout=0.1, rnn_dropout=0.25,
    pretrained_path="scorer_best.bin"
)

scorer.model.load_state_dict(torch.load(best_path, map_location=torch.device('cpu')))
print("Computing test accuracy")
acc = compare(
    scorer,
    "../../data/hypotheses/generated.txt",
    "../../data/gold_analyses/test.csv",
)
