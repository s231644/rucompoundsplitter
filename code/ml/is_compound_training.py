import os
# this command is necessary for reproducibility in Google Colab
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import LSTMIsCompoundClassifier
from vocab import vocab


device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_data(path):
    lemmas, targets = [], []
    with open(path, "r") as f:
        # кругорама	круг:ROOT/о:LINK/рам:ROOT/а:END
        for line in f:
            lemma, morphemes = line.strip().split('\t')
            n_roots = morphemes.count(":ROOT")
            is_compound = n_roots > 1
            lemmas.append(lemma)
            targets.append(is_compound)
    df = pd.DataFrame.from_dict({"lemma": lemmas, "is_compound": targets})
    df["is_compound"] = df["is_compound"].apply(int)
    return df


def collate_fn(data):
    chars = []
    masks = []
    y = []
    data = sorted(data, key=lambda x: -len(x[0]))
    max_len = 0
    for word, y_true in data:
        chars_word = vocab.tokenize(word)
        max_len = max(max_len, len(chars_word))
        chars.append(chars_word)
        y.append([y_true])
    for i in range(len(chars)):
        l = len(chars[i])
        chars[i].extend([vocab.pad_idx] * (max_len - l))
        masks.append([1] * l + [0] * (max_len - l))
    return {
        "chars": torch.tensor(chars, dtype=torch.int64, device=device),
        "masks": torch.tensor(masks, dtype=torch.bool, device=device),
        "y": torch.tensor(y, dtype=torch.float32, device=device)
    }


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

train_df = prepare_data("../../data/tikhonov/train_Tikhonov_reformat.txt")
test_df = prepare_data("../../data/tikhonov/test_Tikhonov_reformat.txt")

model = LSTMIsCompoundClassifier(
    len(vocab.i2w), 128, 256, emb_dropout=0.1, rnn_dropout=0.25
).to(device)

optimizer = model.configure_optimizers()

N_EPOCHS = 30
BATCH_SIZE = 128

val_size = int(len(train_df) * 0.1)

train_dl = DataLoader(train_df.values[:-val_size], batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(train_df.values[-val_size:], batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
test_dl = DataLoader(test_df.values, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

best_val_f1 = 0
savepath = "../../data/models/is_compound_clf_lstm_30_best.bin"

# training
for epoch in range(N_EPOCHS):
    model.train()
    train_losses = []
    for i, batch in tqdm(enumerate(train_dl)):
        loss = model.training_step(batch)
        train_losses.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        val_losses = []
        val_tp, val_fp, val_fn = 0, 0, 0
        for i, batch in tqdm(enumerate(val_dl)):
            loss, tp, fp, fn = model.validation_step(batch)
            val_tp += tp
            val_fp += fp
            val_fn += fn
            val_losses.append(loss.cpu().numpy())
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    precision = val_tp / (val_tp + val_fp)
    recall = val_tp / (val_tp + val_fn)
    f1 = 2 * precision * recall / (precision + recall)
    # Epoch=0: train_loss=0.28297723549063625 val_loss=0.1932332247757075 ...
    print(
        f"Epoch={epoch}: "
        f"train_loss={train_loss} val_loss={val_loss} "
        f"tp={val_tp} fp={val_fp} fn={val_fn} "
        f"P={precision} R={recall} F1={f1}"
    )
    if f1 > best_val_f1:
        best_val_f1 = f1
        savepath = f"is_compound_clf_lstm_{N_EPOCHS}_ckpt_{epoch}.bin"
        torch.save(model.state_dict(), savepath)

# saving
model.load_state_dict(torch.load(savepath, map_location="cpu"))
model.eval()
model.to("cpu")
# torch.save(model.state_dict(), f"is_compound_clf_lstm_{N_EPOCHS}_best.bin")


# error analysis
def error_analysis_step(model, batch):
    chars = batch["chars"]
    masks = batch["masks"]
    y = batch["y"]

    probs = model.forward(chars, masks)

    y_pred = probs >= 0.5
    y_gt = y >= 0.5

    tp = (y_gt * y_pred).reshape(-1)
    fp = ((y_gt != y_pred) * y_pred).reshape(-1)
    fn = ((y_gt != y_pred) * y_gt).reshape(-1)

    return ([vocab.detokenize(x) for i, x in enumerate(chars) if arr[i]] for arr in [tp, fp, fn])


model.to(device)
test_tps, test_fps, test_fns = [], [], []
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dl)):
        batch_tps, batch_fps, batch_fns = error_analysis_step(model, batch)
        test_tps.extend(batch_tps)
        test_fps.extend(batch_fps)
        test_fns.extend(batch_fns)
print(len(test_tps), len(test_fps), len(test_fns))
print("Test false positives", test_fps)
print("Test false negatives", test_fns)
