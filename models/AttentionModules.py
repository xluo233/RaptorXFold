#!/usr/bin/env python3
# encoding: utf-8

from IPython.core.debugger import set_trace
import torch
from torch import nn
import torch.nn.functional as F

import math
import itertools


class SeqWeightAttention(nn.Module): 
    def __init__(self, in_dim, out_dim, n_head, dropout=0.0, *args, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.head_dim = out_dim // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(in_dim, out_dim, bias=False)
        self.to_k = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, MSA_emb, seq_mask):
        # seq_mask: (n_batch, n_row)
        n_batch, n_row, n_col, _ = MSA_emb.size()

        MSA_emb = MSA_emb.permute(0,2,1,3) # (n_batch, n_col, n_row, _)
        query_emb = MSA_emb[:,:,0].unsqueeze(2) # (n_batch, n_col, 1, _)

        Q = self.to_q(query_emb).view(n_batch, n_col, 1, self.n_head, self.head_dim).permute(0,1,3,2,4).contiguous() # (n_batch, n_col, n_head, 1, _)
        K = self.to_k(MSA_emb).view(n_batch, n_col, n_row, self.n_head, self.head_dim).permute(0,1,3,4,2).contiguous() # (n_batch, n_col, n_head, _, n_row)

        Q = Q * self.scale
        QK = torch.matmul(Q, K) # (n_batch, n_col, n_head, 1, n_row)
        QK.masked_fill_(seq_mask[:, None, None, None, :]==0, torch.finfo(QK.dtype).min)
        attn = QK.softmax(dim=-1) # there is a bug in SoftmaxBackward when n_row==1
        attn = attn.permute(0,4,1,2,3).contiguous() # (n_batch, n_row, n_col, n_head, 1, )
        return self.dropout(attn)


class SelfAttention(nn.Module): 
    '''
    Self Attention Class for Transformer.
    '''
    def __init__(self,
                    in_dim: int, # emb size
                    out_dim: int, # out emb size
                    n_head: int, # head num
                    scaling: int, # scaling value
                    seqlen_scaling: bool=True, # use seq len as scaling
                    attn_seq_weight: bool=False, # use SeqWeightAttention module
                    attn_dropout: float=0.0, # attn dropout
                    mean_attn: bool=True, # return mean of attn
                    *args, **kwargs
                ):
        super().__init__()
        self.n_head = n_head
        self.scaling = scaling
        self.seqlen_scaling = seqlen_scaling
        self.attn_seq_weight = attn_seq_weight 
        self.mean_attn = mean_attn 

        self.to_q = nn.Linear(in_dim, out_dim, bias=False)
        self.to_k = nn.Linear(in_dim, out_dim, bias=False)
        self.to_v = nn.Linear(in_dim, out_dim, bias=False)

        if self.attn_seq_weight: 
            self.seq_weight_attn_layer = SeqWeightAttention(in_dim, out_dim, n_head, attn_dropout)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None):
        # x: [n_batch, n_row, n_col, n_head*n_emb]
        n_batch, n_row, n_col, n_head_x_n_emb = x.size()
        scaling = self.scaling / math.sqrt(n_col) if self.seqlen_scaling else self.scaling

        # Q, K, V: [n_batch, n_row, n_col, n_head, n_emb]
        Q = self.to_q(x).view(n_batch, n_row, n_col, self.n_head, -1)
        K = self.to_k(x).view(n_batch, n_row, n_col, self.n_head, -1)
        V = self.to_v(x).view(n_batch, n_row, n_col, self.n_head, -1)

        Q = Q * scaling
        if self.attn_seq_weight:
            attn_seq_weight = self.seq_weight_attn_layer(x, seq_mask)
            Q = Q * attn_seq_weight
            QK = torch.einsum(f"nrihd,nrjhd->nhij", Q, K).unsqueeze(2) # [n_batch, n_head, 1, n_col, n_col]
        else:
            QK = torch.einsum(f"nrihd,nrjhd->nhrij", Q, K) # [n_batch, n_head, n_row, n_col, n_col]

        if res_mask is not None:
            res_mask = res_mask[:, :, None] * res_mask[:, None, :]
            QK.masked_fill_(res_mask[:, None, None, :, :]==0, torch.finfo(QK.dtype).min)

        # attn, [n_batch, n_head, n_row, n_col, n_col]
        attn = QK.softmax(-1)

        # dropout
        attn = self.attn_dropout(attn)

        # out
        if self.attn_seq_weight:
            out = torch.einsum(f"nhij,nrjhd->nrihd", attn.squeeze(2), V)
        else:
            out = torch.einsum(f"nhrij,nrjhd->nrihd", attn, V)

        out = out.contiguous().view(n_batch, n_row, n_col, -1)


        return out

class MHSelfAttention(nn.Module):
    '''
    Multihead attention on row (i.e. residues in each sequence), with each head being a Fast Attention Head.
    Agrs:
        emb_dim (int): the embedding dim.
        n_head (int): the head num.
        head_by_head (bool): True: calc each head one after another, False: calc all heads simultaneously.
    '''
    def __init__(self,
                    emb_dim: int,
                    n_head: int,
                    out_dropout: float=0.0,
                    attn_dropout: float=0.0,
                    seqlen_scaling: bool=False,
                    head_by_head: bool=True,
                    mean_attn: bool=True,
                    attn_seq_weight: bool=False,
                    *args, **kwargs
                ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_dim // n_head
        self.attn_dropout = attn_dropout
        self.seqlen_scaling = seqlen_scaling
        self.head_by_head = head_by_head


        self.mean_attn = mean_attn #attn_config['mean_attn']
        self.attn_seq_weight = attn_seq_weight #attn_config['attn_seq_weight']

        self.scaling = self.head_dim ** -0.5

        # calc each head one after another
        if  self.head_by_head:
            self.heads = nn.ModuleList()
            for i in range(n_head):
                self.heads.append(self.get_attn_module(emb_dim, self.head_dim, 1))
        # calc all heads simultaneously
        else:
            self.heads = self.get_attn_module(emb_dim, emb_dim, n_head)

        self.to_out = nn.Linear(emb_dim, emb_dim, bias=True)
        self.out_dropout = nn.Dropout(out_dropout)

    def get_attn_module(self, in_dim, out_dim, n_head):
        return SelfAttention(in_dim, out_dim, n_head, scaling=self.scaling, seqlen_scaling=self.seqlen_scaling, attn_dropout=self.attn_dropout, attn_seq_weight=self.attn_seq_weight, mean_attn=self.mean_attn )


    def forward(self, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None):

        # calc each head one after another
        if  self.head_by_head:
            out, attn = [], []
            for i in range(self.n_head):
                head = self.heads[i]
                head_out = head(x, seq_weight, res_mask, seq_mask, pair_feat)

                out.append(head_out)

            # concat outs, attns
            out = torch.cat(out, dim=-1)

        # calc all heads simultaneously
        else:
            heads_out = self.heads(x, seq_weight, res_mask, seq_mask, pair_feat)
            out = heads_out

        out = self.to_out(out)
        out = self.out_dropout(out)


        return out

class PointwiseAttention(nn.Module): 
    '''
    TemplatePointwiseAttention Class for Template Embedding
    '''
    def __init__(self, key_dim=128, value_dim=64, num_head=4, out_dim=64):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim = key_dim // num_head
        self.q_weights = nn.Parameter(
            torch.zeros(key_dim, num_head, key_dim // num_head))
        self.k_weights = nn.Parameter(
            torch.zeros(value_dim, num_head, key_dim // num_head))
        self.v_weights = nn.Parameter(
            torch.zeros(value_dim, num_head, value_dim // num_head))

        self.o_weights = nn.Parameter(
            torch.zeros(num_head, value_dim // num_head, out_dim))
        self.o_bias = nn.Parameter(
            torch.zeros(out_dim))

        # initialize the weight
        nn.init.xavier_uniform_(self.q_weights)
        nn.init.xavier_uniform_(self.k_weights)
        nn.init.xavier_uniform_(self.v_weights)

    # q_data: A tensor of queries, (B * L * L, 1, C1)
    # m_data: A tensor of memoies from which the keys and values are projected,
    #   (B * L * L, T, C2)
    # mask: bias for the attention, (B*L*L, T)
    def forward(self, q_data, m_data, mask):
        Q = torch.einsum(
            'bqa,ahc->bqhc', q_data, self.q_weights) * self.key_dim**(-0.5)
        K = torch.einsum(
            'bka,ahc->bkhc', m_data, self.k_weights)
        V = torch.einsum(
            'bka,ahc->bkhc', m_data, self.v_weights)

        logits = torch.einsum('bqhc,bkhc->bhqk', Q, K)
        logits.masked_fill_(mask[:, None, None, :] == 0,
                            torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        weight_avg = torch.einsum('bhqk,bkhc->bqhc', weights, V)

        output = torch.einsum(
            'bqhc,hco->bqo', weight_avg, self.o_weights) + self.o_bias
        return output
