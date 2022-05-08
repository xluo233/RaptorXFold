#!/usr/bin/env python3
# encoding: utf-8

from torch import nn

from .Common import FeedForward, Residual

from .AttentionModules import MHSelfAttention

from IPython.core.debugger import set_trace

class AxialTransformerLayer(nn.Module):
    def __init__(self, emb_dim, n_head, 
                        out_dropout=0.0, attn_dropout=0.0, seqlen_scaling=True, 
                        include_ff=True, n_ff_hidden=None, activation='elu', ff_dropout=0.0, 
                        pre_layernorm=True, head_by_head=True,
                    *args, **kwargs):
        super().__init__()
        assert emb_dim % n_head == 0, f"embedding dim ({emb_dim}) must be divisible by head num ({n_head})."

        self.include_ff = include_ff

        # row attn
        row_attn_layer = MHSelfAttention(emb_dim, n_head=n_head,  mean_attn=False, attn_seq_weight=False, 
                            seqlen_scaling=seqlen_scaling, out_dropout=out_dropout, attn_dropout=attn_dropout, 
                            head_by_head=head_by_head)
        self.row_attn_layer = Residual(row_attn_layer, emb_dim, emb_dim, pre_layernorm=pre_layernorm)

        # col attn
        col_attn_layer = MHSelfAttention(emb_dim, n_head=n_head, mean_attn=False, attn_seq_weight=False, 
                            seqlen_scaling=seqlen_scaling, out_dropout=out_dropout, attn_dropout=attn_dropout, 
                            head_by_head=head_by_head)
        self.col_attn_layer = Residual(col_attn_layer, emb_dim, emb_dim, pre_layernorm=pre_layernorm)

        # feed forward
        ff_layer = FeedForward(emb_dim, emb_dim, n_hidden=n_ff_hidden, activation=activation, dropout=ff_dropout)
        self.ff_layer = Residual(ff_layer, emb_dim, emb_dim, pre_layernorm=pre_layernorm)

    def forward(self, x, attn_mask=None, *args, **kwargs):
        # x: [n_batch, n_row, n_col, n_head*n_emb]

        x = self.row_attn_layer(x, attn_mask) # [n_batch, n_col, n_row, n_head*n_emb]
        x = self.col_attn_layer(x.transpose(1,2), attn_mask).transpose(1,2) # [n_batch, n_row, n_col, n_head*n_emb]
        x = self.ff_layer(x)
        
        return x
