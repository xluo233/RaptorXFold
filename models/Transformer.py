#!/usr/bin/env python3
# encoding: utf-8

import copy
import torch
from torch import nn

from .AttentionModules import MHSelfAttention
from .Common import FeedForward, Residual

from IPython.core.debugger import set_trace

class TransformerLayer(nn.Module): 
    def __init__(self, emb_dim, n_head, 
                        out_dropout=0.0, attn_dropout=0.0, seqlen_scaling=True, 
                        n_ff_hidden=None, activation='elu', ff_dropout=0.0, 
                        pre_layernorm=True, head_by_head=True,
                    *args, **kwargs):
        super().__init__()
        assert emb_dim % n_head == 0, f"embedding dim ({emb_dim}) must be divisible by head num ({n_head})."

        self.head_dim = emb_dim // n_head


        attn_row = MHSelfAttention(emb_dim, n_head, mean_attn=True, attn_seq_weight=True, 
                                    seqlen_scaling=seqlen_scaling, out_dropout=out_dropout, attn_dropout=attn_dropout,
                                    head_by_head=head_by_head)
        self.attn_row = Residual(attn_row, emb_dim, emb_dim, pre_layernorm=pre_layernorm)


        attn_col = MHSelfAttention(emb_dim, n_head, mean_attn=True,  attn_seq_weight=False,
                                    seqlen_scaling=seqlen_scaling, out_dropout=out_dropout, attn_dropout=attn_dropout, 
                                    head_by_head=head_by_head)
        self.attn_col = Residual(attn_col, emb_dim, emb_dim, pre_layernorm=pre_layernorm)


        if n_ff_hidden is None: n_ff_hidden = emb_dim
        ff_layer = FeedForward(emb_dim, emb_dim, n_hidden=n_ff_hidden, activation=activation, dropout=ff_dropout)
        self.ff_layer = Residual(ff_layer, emb_dim, emb_dim, pre_layernorm=pre_layernorm)

    def get_head_dim(self, ):
        return self.head_dim

    def forward(self, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None, *args, **kwargs):

        x = self.attn_row(x, seq_weight, res_mask, seq_mask, pair_feat)
        x = self.attn_col(x.transpose(1,2), seq_weight, seq_mask).transpose(1,2)
        x = self.ff_layer(x)
        return x
