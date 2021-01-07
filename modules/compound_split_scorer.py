import dynet as dy
from typing import Optional, List

from modules.embedders import BaseEmbedder, CharEmbedderPOS
from modules.encoders import BiLSTMEncoder
from modules.layers import Attention


class CompoundSplitScorer:
    """
    A class for scoring of compound splits
    """
    def __init__(
            self,
            n_vocab: int,
            n_pos: int,
            n_rules: int,
            d_input: int,
            d_pos: int,
            d_rule: int,
            d_hidden: int,
            n_layers: Optional[int] = 1,
            attn_type: Optional[str] = None
    ):
        self.model = dy.Model()
        self.token_embedder = CharEmbedderPOS(
            self.model, n_vocab, n_pos, d_input, d_pos
        )
        self.rule_embedder = BaseEmbedder(self.model, n_rules, d_rule)
        self.token_encoder = BiLSTMEncoder(self.model, d_input + d_rule, d_hidden, n_layers)

        d_rnn_output = d_hidden

        self.attn_type = attn_type

        if self.attn_type in ['att', 'selfatt']:
            self.attention = Attention(self.model, d_rnn_output, d_rnn_output, d_rule)
        if self.attn_type == 'selfatt':
            self.self_attention = Attention(self.model, d_rnn_output, d_rnn_output)

        self.W = self.model.add_parameters((d_rnn_output, d_rule))
        # self.clf = self.model.add_parameters((1, d_rule + d_rnn_output))

    def forward(
            self,
            input_c: List[int],
            pos_c: int,
            input_m: List[int],
            pos_m: int,
            input_h: List[int],
            pos_h: int,
            rule_id: int
    ):
        embedded_m = self.token_embedder.forward(input_m, pos_m)
        encoded_m, _ = self.token_encoder.forward(embedded_m)

        embedded_h = self.token_embedder.forward(input_h, pos_h)
        encoded_h, _ = self.token_encoder.forward(embedded_h)

        embedded_c = self.token_embedder.forward(input_c, pos_c)
        encoded_c, _ = self.token_encoder.forward(embedded_c)

        embedded_rule = self.rule_embedder.embed_token(rule_id)

        encoded = encoded_m + encoded_h + encoded_c

        if self.attn_type == 'selfatt':
            hm = dy.concatenate_cols(encoded)
            hm = self.self_attention.encode(hm)
            hm = self.attention.encode(hm, embedded_rule)
        elif self.attn_type == 'att':
            hm = dy.concatenate_cols(encoded)
            hm = self.attention.encode(hm, embedded_rule)
        else:
            hm = dy.emax(encoded)

        score = self.W * embedded_rule
        score = dy.dot_product(hm, score)
        # score = self.clf * dy.concatenate([hm, embedded_rule])

        return score


class CompoundSplitClf:
    """
    A class for classification of compound splits
    """
    def __init__(
            self,
            n_vocab: int,
            n_pos: int,
            n_rules: int,
            d_input: int,
            d_pos: int,
            d_rule: int,
            d_hidden: int,
            n_layers: Optional[int] = 1,
            attn_type: Optional[str] = None
    ):
        self.model = dy.Model()
        self.token_embedder = CharEmbedderPOS(
            self.model, n_vocab, n_pos, d_input, d_pos
        )
        self.rule_embedder = BaseEmbedder(self.model, n_rules, d_rule)
        self.token_encoder = BiLSTMEncoder(self.model, d_input + d_rule, d_hidden, n_layers)

        d_rnn_output = d_hidden

        self.attn_type = attn_type

        if self.attn_type in ['att', 'selfatt']:
            self.attention = Attention(self.model, d_rnn_output, d_rnn_output, d_rule)
        if self.attn_type == 'selfatt':
            self.self_attention = Attention(self.model, d_rnn_output, d_rnn_output)

        # self.W = self.model.add_parameters((d_rnn_output, d_rule))
        self.clf = self.model.add_parameters((1, d_rule + d_rnn_output))

    def forward(
            self,
            input_c: List[int],
            pos_c: int,
            input_m: List[int],
            pos_m: int,
            input_h: List[int],
            pos_h: int,
            rule_id: int
    ):
        embedded_m = self.token_embedder.forward(input_m, pos_m)
        embedded_h = self.token_embedder.forward(input_h, pos_h)
        encoded_hm, _ = self.token_encoder.forward(embedded_m + embedded_h)

        embedded_c = self.token_embedder.forward(input_c, pos_c)
        encoded_c, _ = self.token_encoder.forward(embedded_c)

        embedded_rule = self.rule_embedder.embed_token(rule_id)

        encoded = encoded_hm + encoded_c

        if self.attn_type == 'selfatt':
            hm = dy.concatenate_cols(encoded)
            hm = self.self_attention.encode(hm)
            hm = self.attention.encode(hm, embedded_rule)
        elif self.attn_type == 'att':
            hm = dy.concatenate_cols(encoded)
            hm = self.attention.encode(hm, embedded_rule)
        else:
            hm = dy.emax(encoded)

        # score = self.W * embedded_rule
        # score = dy.dot_product(hm, score)
        score = self.clf * dy.concatenate([hm, embedded_rule])

        return score
