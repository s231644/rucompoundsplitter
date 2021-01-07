import dynet as dy
import numpy as np

from typing import List


class BaseEmbedder:
    def __init__(
            self,
            model,
            n_vocab: int,
            d_input: int
    ):
        self.model = model

        self.n_vocab = n_vocab
        self.d_input = d_input

        self.emb = self.model.add_lookup_parameters(
            (self.n_vocab, self.d_input),
            init='uniform',
            scale=1 / np.sqrt(self.d_input)
        )

    def embed_token(self, token: int):
        return self.emb[token]

    def forward(self, tokens: List[int]) -> List[dy.FloatVectorValue]:
        return [self.embed_token(token) for token in tokens]


class CharEmbedderPOS:
    def __init__(
            self,
            model,
            n_vocab: int,
            n_pos,
            d_input: int,
            d_pos: int
    ):
        self.model = model

        self.token_embedder = BaseEmbedder(self.model, n_vocab, d_input)
        self.n_pos = n_pos
        self.d_pos = d_pos

        self.pos_emb = self.model.add_lookup_parameters(
            (self.n_pos, self.d_pos),
            init='uniform',
            scale=1 / np.sqrt(self.d_pos)
        )

    def forward(
            self, tokens: List[int],
            pos_token: int,
    ) -> List[dy.FloatVectorValue]:
        pos_embed = self.pos_emb[pos_token]
        return [
            dy.concatenate([self.token_embedder.embed_token(token), pos_embed])
            for token in tokens
        ]
