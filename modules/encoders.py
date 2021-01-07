import dynet as dy
from typing import List


class BiLSTMEncoder:
    def __init__(
            self,
            model: dy.Model,
            d_input: int,
            d_hidden: int,
            n_layers: int = 1
    ):
        self.model = model

        self.rnn = dy.BiRNNBuilder(
            n_layers, d_input, d_hidden, self.model, dy.VanillaLSTMBuilder
        )

    def forward(self, embeddings: List[dy.FloatVectorValue]):
        rnn_outputs = self.rnn.add_inputs(embeddings)
        hs, cs = [], []
        for fw, bw in rnn_outputs:
            hs.append(dy.concatenate([fw.s()[0], bw.s()[0]]))
            cs.append(dy.concatenate([fw.s()[1], bw.s()[1]]))
        return hs, cs
