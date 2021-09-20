import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from overrides import overrides


class LSTMIsCompoundClassifier(nn.Module):
    def __init__(
            self,
            n_vocab: int,
            d_input: int,
            d_rnn: int,
            n_layers: int = 2,
            emb_dropout: float = 0.0,
            rnn_dropout: float = 0.0
    ):
        super().__init__()
        self.char_embedder = nn.Embedding(n_vocab, d_input)
        self.embedding_dropout = nn.Dropout(emb_dropout)

        self.rnn = nn.LSTM(
            d_input, d_rnn,
            num_layers=n_layers, bidirectional=True,
            batch_first=True, dropout=rnn_dropout
        )
        self.d_rnn_output = 2 * n_layers * d_rnn

        self.context2query = nn.Linear(self.d_rnn_output, 2 * d_rnn)
        self.attn = nn.MultiheadAttention(2 * d_rnn, 1)
        self.act = nn.ReLU()

        self.clf = nn.Linear(2 * d_rnn, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.BCELoss()

    def forward(self, chars, padding_mask):
        # chars --- batch_size x seq_len
        # padding_mask --- batch_size x seq_len

        # batch_size x seq_len x d_input
        chars_embed = self.char_embedder(chars)
        packed_input = pack_padded_sequence(
            self.embedding_dropout(chars_embed),
            padding_mask.sum(1).cpu().numpy(),
            batch_first=True
        )
        rnn_outputs, (h, c) = self.rnn(packed_input)
        rnn_outputs, input_sizes = pad_packed_sequence(
            rnn_outputs, batch_first=True
        )
        # seq_len x batch_size x d_rnn
        rnn_outputs = rnn_outputs.transpose(0, 1)
        # c: 2 * n_layers x batch_size x d_rnn
        context = c.transpose(0, 1).reshape(1, -1, self.d_rnn_output)
        query = self.context2query(context)
        attention, _ = self.attn(
            query, rnn_outputs, rnn_outputs,
            key_padding_mask=~padding_mask
        )
        attention = self.act(attention).squeeze(0)
        output = self.sigmoid(self.clf(attention))
        return output

    def training_step(self, batch):
        chars = batch["chars"]
        masks = batch["masks"]
        y = batch["y"]

        probs = self.forward(chars, masks)
        loss = self.loss_fn(probs, y)

        return loss

    def validation_step(self, batch):
        chars = batch["chars"]
        masks = batch["masks"]
        y = batch["y"]

        probs = self.forward(chars, masks)
        loss = self.loss_fn(probs, y)

        y_pred = probs >= 0.5
        y_gt = y >= 0.5

        tp = ((y_gt == y_pred) * y_pred).sum()
        fp = ((y_gt != y_pred) * y_pred).sum()
        fn = ((y_gt != y_pred) * y_gt).sum()

        return loss, tp, fp, fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class LSTMHypothesesClassifier(LSTMIsCompoundClassifier):
    def __init__(
            self,
            n_vocab: int,
            n_rules: int,
            d_input: int,
            d_rnn: int,
            n_layers: int = 2,
            emb_dropout: float = 0.0,
            rnn_dropout: float = 0.0
    ):
        super().__init__(
            n_vocab=n_vocab,
            d_input=d_input,
            d_rnn=d_rnn,
            n_layers=n_layers,
            emb_dropout=emb_dropout,
            rnn_dropout=rnn_dropout
        )
        self.rule_embedder = nn.Embedding(n_rules, self.d_rnn_output)
        self.part_embedder = nn.Embedding(3, self.d_rnn_output)  # c, m, h

        self.context_part_emb = nn.Sequential(
            nn.Linear(self.d_rnn_output, d_rnn),
            nn.Tanh(),
            nn.Linear(d_rnn, self.d_rnn_output),
            nn.ReLU()
        )
        self.analysis_clf = nn.Linear(2 * d_rnn, 1)  # is correct

    @staticmethod
    def invert_permutation(permutation):
        output = torch.empty_like(permutation)
        output.scatter_(
            0, permutation, torch.arange(
                0, len(permutation), dtype=torch.int64,
                device=permutation.device
            )
        )
        return output

    def encode_c(self, chars):
        chars = chars.unsqueeze(0)
        # 1 x seq_len_c x d_input
        chars_embed = self.embedding_dropout(self.char_embedder(chars))
        # c: 2 * n_layers x 1 x d_rnn
        rnn_outputs, (h, c) = self.rnn(chars_embed)
        # seq_len x 1 x d_rnn
        rnn_outputs = rnn_outputs.transpose(0, 1)
        # 1 x 1 x 2 * n_layers * d_rnn
        context = c.transpose(0, 1).reshape(1, -1, self.d_rnn_output)
        return rnn_outputs, context

    def encode_m_or_h(self, chars, padding_mask):
        # N x seq_len x d_input
        chars_embed = self.embedding_dropout(self.char_embedder(chars))
        lens = padding_mask.sum(1)
        inds = lens.argsort(descending=True)
        inds_inv = self.invert_permutation(inds)
        packed_input = pack_padded_sequence(
            chars_embed[inds],
            lens[inds].cpu().numpy(),
            batch_first=True
        )
        rnn_outputs, (h, c) = self.rnn(packed_input)
        rnn_outputs, input_sizes = pad_packed_sequence(
            rnn_outputs, batch_first=True
        )
        # permute back
        rnn_outputs = rnn_outputs[inds_inv]
        # seq_len x N x d_rnn
        rnn_outputs = rnn_outputs.transpose(0, 1)
        # 1 x N x 2 * n_layers * d_rnn
        context = c.transpose(0, 1).reshape(1, -1, self.d_rnn_output)
        return rnn_outputs, context

    @staticmethod
    def cat_rnn_outputs(
            rnn_outputs_c, rnn_outputs_ms, rnn_outputs_hs,
            padding_mask_ms, padding_mask_hs, n
    ):
        rnn_outputs = torch.cat([
            rnn_outputs_c.repeat(1, n, 1), rnn_outputs_ms, rnn_outputs_hs
        ])
        padding_mask_cs = torch.ones(
            (n, len(rnn_outputs_c)), dtype=torch.bool, device=rnn_outputs_c.device
        )
        padding_mask = torch.cat([
            padding_mask_cs, padding_mask_ms, padding_mask_hs
        ], dim=-1)
        return rnn_outputs, padding_mask

    @overrides
    def forward(
            self,
            chars_c,
            chars_ms, chars_hs,
            rule_ids_c, rule_ids_m, rule_ids_h,
            padding_mask_ms, padding_mask_hs,
    ):
        # chars_c --- seq_len_c
        # chars_ms --- N x seq_len_m
        # chars_hs --- N x seq_len_h
        # rule_ids_c, rule_ids_m, rule_ids_h --- N, N, N
        # padding_mask_ms --- N x seq_len_m
        # padding_mask_hs --- N x seq_len_h

        n = len(rule_ids_c)

        # processing c
        rnn_outputs_c, context_c = self.encode_c(chars_c)
        # rnn_outputs_c --- seq_len_c x 1 x d_rnn
        # context_c --- 1 x 1 x 2 * n_layers * d_rnn

        # processing m
        rnn_outputs_ms, context_ms = self.encode_m_or_h(chars_ms, padding_mask_ms)
        # rnn_outputs_m --- seq_len_m x N x d_rnn
        # context_m --- 1 x N x 2 * n_layers * d_rnn

        # processing h
        rnn_outputs_hs, context_hs = self.encode_m_or_h(chars_hs, padding_mask_hs)
        # rnn_outputs_h --- seq_len_h x N x d_rnn
        # context_h --- 1 x N x 2 * n_layers * d_rnn

        extended_context_c = self.context_part_emb(
            context_c + self.rule_embedder(rule_ids_c) + self.part_embedder(
                torch.tensor([[0]], device=context_c.device)
            )
        )
        extended_context_ms = self.context_part_emb(
            context_ms + self.rule_embedder(rule_ids_m) + self.part_embedder(
                torch.tensor([[1] * n], device=context_ms.device)
            )
        )
        extended_context_hs = self.context_part_emb(
            context_hs + self.rule_embedder(rule_ids_h) + self.part_embedder(
                torch.tensor([[2] * n], device=context_hs.device)
            )
        )
        context = extended_context_c + extended_context_ms + extended_context_hs  # 1 x N x 2 * n_layers * d_rnn
        query = self.context2query(context)  # 1 x N x 2 * d_rnn
        rnn_outputs, padding_mask = self.cat_rnn_outputs(
            rnn_outputs_c, rnn_outputs_ms, rnn_outputs_hs,
            padding_mask_ms, padding_mask_hs, n
        )
        attention, _ = self.attn(query, rnn_outputs, rnn_outputs, key_padding_mask=~padding_mask)
        attention = self.act(attention).squeeze(0)
        output = self.sigmoid(self.analysis_clf(attention))  # N x 1

        return output
