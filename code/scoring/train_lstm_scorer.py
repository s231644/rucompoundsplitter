import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from code.ml.vocab import vocab
from scorer_abstract import HypothesesSample
from scorer_lstm import LSTMScorer
from evaluate import read_analyses, read_gold, compute_metrics, matched

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

best_path = "scorer_best.bin"


def train(
        scorer: LSTMScorer,
        preds_path: str,
        train_path: str,
        val_path: Optional[str] = None,
        n_epochs: int = 1,
        batch_size: int = 8,
        device: str = "cpu"
):
    preds = read_analyses(preds_path)
    golds_train = read_gold(train_path)

    samples = []
    labels = []

    for k in tqdm(golds_train):
        word_c, pos_c = k
        a_gold = golds_train[k]
        a_preds = preds.get(k, [])
        correct_ids = [
            i for i, a_pred in enumerate(a_preds) if matched(a_gold, a_pred)
        ]
        if not correct_ids:
            labels.append(len(a_preds))
            a_preds.append(a_gold)
        else:
            labels.append(correct_ids[0])
        h = HypothesesSample(word_c, pos_c, a_preds)
        samples.append(h)

    scorer.model.to(device)
    best_acc = -1
    best_loss = 1e6

    optimizer = scorer.model.configure_optimizers()
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        scorer.model.train()
        inds = np.arange(len(samples))
        np.random.shuffle(inds)

        epoch_losses = []
        loss = 0
        for i in tqdm(inds):
            sample = samples[inds[i]]
            label = labels[inds[i]]

            batch = scorer.prepare_model_input(
                sample.word_c, sample.pos_c, sample.analyses
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            res = scorer.model(**batch)  # N x 1
            target = torch.zeros_like(res, requires_grad=False, device=device)
            target[label] = 1
            loss += scorer.loss_fn(res, target)
            if (i + 1) % batch_size == 0 or i + 1 == len(samples):
                loss /= batch_size
                loss.backward()
                epoch_losses.append(loss.detach().cpu().numpy())
                optimizer.step()
                optimizer.zero_grad()
                loss = 0
        scorer.model.eval()
        mean_loss = np.mean(epoch_losses)
        print(
            f"Epoch {epoch} "
            f"train_loss={mean_loss}"
        )
        if val_path:
            scorer.model.to("cpu")
            acc = compute_metrics(scorer, preds_path, val_path)
            print(f"val_acc={acc}")
            if acc > best_acc:
                best_acc = acc
                torch.save(scorer.model.state_dict(), best_path)
            scorer.model.to(device)
        # if mean_loss < best_loss:
        #     best_loss = mean_loss
        #     torch.save(scorer.model.state_dict(), best_path)
    return best_path


if __name__ == "__main__":
    scorer = LSTMScorer(
        len(vocab.i2w), 250, 128, 256, emb_dropout=0.1, rnn_dropout=0.25,
        pretrained_path="../../data/models/is_compound_clf_lstm_30_best.bin"
    )

    best_path = train(
        scorer,
        "../../data/hypotheses/generated.txt",
        "../../data/gold_analyses/train.csv",
        "../../data/gold_analyses/val.csv",
        n_epochs=1, batch_size=8, device="cpu"
    )
    #
    # scorer.model.to("cpu")
    # scorer.model.load_state_dict(torch.load(best_path))
    #
    # print("Computing valid accuracy")
    # acc = compute_metrics(
    #     scorer,
    #     "../../data/hypotheses/generated.txt",
    #     "../../data/gold_analyses/val.csv",
    # )

    scorer.model.to("cpu")
    scorer.model.eval()
    print("Computing test accuracy")
    acc = compute_metrics(
        scorer,
        "../../data/hypotheses/generated.txt",
        "../../data/gold_analyses/test.csv",
    )
