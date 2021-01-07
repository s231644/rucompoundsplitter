import dynet as dy
import numpy as np


class Attention(object):
    """
    A module for both self attention and normal attention with key, query and value
    """
    def __init__(self, model, dm, dk, dq=None):
        # dm = memory dimension
        # dk = key dimension
        # dq = query dimension (None for self-attention)
        dq = dq or dm
        self.w_q = model.add_parameters((dk, dq))
        self.w_k = model.add_parameters((dk, dm))
        self.w_v = model.add_parameters((dk, dm))
        self.factor = dk ** 0.5

    def encode(self, memory, query=None):
        query = query or memory  # if no query then self attention
        Q = self.w_q * query
        K = self.w_k * memory
        V = self.w_v * memory
        A = dy.softmax(dy.transpose(K) * Q / self.factor)
        out = V * A
        return out

