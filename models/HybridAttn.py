#!/usr/bin/env python3
# encoding: utf-8

import math
import torch
from torch import nn

from .Common import FeedForward, Residual, MSAOutputModule

from .ResNet2D import ResNet2DBlock
from .AxialTransformer import AxialTransformerLayer
from .AttentionModules import SeqWeightAttention
from .Transformer import TransformerLayer

from IPython.core.debugger import set_trace

class CovLayer(nn.Module):
    '''
    Covariance layer.
    '''
    def __init__(self, n_emb_1D=32, n_emb_2D=64, layer_norm_pair_feat=False, *args, **kwargs):
        super().__init__()
        self.layer_norm_pair_feat = layer_norm_pair_feat
        if layer_norm_pair_feat: self.norm_layer = nn.LayerNorm(n_emb_1D**2)
        self.proj_layer = nn.Linear(n_emb_1D**2, n_emb_2D, bias=True)

    def calc_pair_outer_product(self, x_left, x_right, weight):
        n_batch, seq_num, n_res, emb_dim = x_left.shape

        x_left = x_left * weight
        outer_product = torch.einsum('bnil,bnjr->bijlr', x_left, x_right)
        outer_product = outer_product.reshape(n_batch, n_res, n_res, -1)

        return outer_product

    def forward(self, x_left, x_right, seq_weight, *args, **kwargs):
        # x_left, x_right: n_batch, seq_num, n_res, emb_dim
        # outer product: n_batch, seq_len, seq_len, out_emb_2D
        outer_product = self.calc_pair_outer_product(x_left, x_right, seq_weight)
        if self.layer_norm_pair_feat: outer_product = self.norm_layer(outer_product)
        pair_feat = self.proj_layer(outer_product)

        return pair_feat

class MSA_to_Pair(nn.Module):
    '''
    Generate initial pairwise feature from MSA feature.
    '''
    def __init__(self, emb_dim=512, n_emb_1D=32, n_emb_2D=64, n_ResNet2D_block=1,
                    seq_weight_attn_dropout=0.0, # dropout for SeqWeightAttention
                    activation='elu', normalization='instance', bias=False, dropout=0.0, # ResNet2DBlock

                    *args, **kwargs
                ):
        super().__init__()
        self.n_emb_2D = n_emb_2D
        self.n_ResNet2D_block = n_ResNet2D_block

        self.layer_norm_MSA = nn.LayerNorm(emb_dim)

        # 1D feat
        self.proj_1D_left = nn.Sequential(
            nn.Linear(emb_dim, n_emb_1D, bias=True),
            nn.LayerNorm(n_emb_1D)
        )

        #if self.seq_weight_type == 'attn_weight':
        self.seq_weight_attn_layer = SeqWeightAttention(n_emb_1D, n_emb_1D, 1, seq_weight_attn_dropout)

        # outer product
        self.cov_layer = CovLayer(n_emb_1D, n_emb_2D, bias)

        # layer_norm on 2D feat
        self.norm_pair1 = nn.LayerNorm(n_emb_2D)
        self.norm_pair2 = nn.LayerNorm(n_emb_2D)

        n_in_2D = n_emb_2D
        # 1D feat dim
        n_in_2D += n_emb_1D*2

        # previous 2D feature dim
        n_in_2D += n_emb_2D

        self.proj_2D = nn.Sequential(
            nn.Conv2d(n_in_2D, n_emb_2D, kernel_size=1, bias=True),
            nn.InstanceNorm2d(n_emb_2D)
        )

        # ResNet2D Blocks
        self.ResNet2D_blocks = nn.ModuleList(
            [
                ResNet2DBlock(
                    n_emb_2D, n_emb_2D, kernel_size=3, dilation=1, dropout=dropout, activation=activation, normalization=normalization, bias=bias
                ) for _ in range(n_ResNet2D_block)
            ]
        )

    def get_out_channel(self, ):
        return self.n_emb_2D

    def get_pair_feat_1D(self, x_left, x_right, seq_weight):
        seq_len = x_left.shape[-2]

        x_left = (x_left * seq_weight).sum(1)
        x_right = (x_right * seq_weight).sum(1)

        x_left = x_left.unsqueeze(1).repeat(1, seq_len, 1, 1)
        x_right = x_right.unsqueeze(2).repeat(1, 1, seq_len, 1)

        return torch.cat((x_left, x_right), dim=-1)

    def forward(self, x, seq_weight=None, seq_mask=None, pair_feat_prev=None, attn=None, **kwargs):
        # x: [n_batch, seq_num, seq_len, emb_dim]

        x = self.layer_norm_MSA(x)

        # 1D project down
        feat_1D_left = self.proj_1D_left(x)
        feat_1D_right = feat_1D_left


        seq_weight = self.seq_weight_attn_layer(feat_1D_left, seq_mask).squeeze(-1)

        # outer product: n_batch, seq_len, seq_len, n_emb_2D
        pair_feat = self.cov_layer(feat_1D_left, feat_1D_right, seq_weight)

        # norm pair feat
        pair_feat = self.norm_pair1(pair_feat)
        pair_feat_prev = self.norm_pair2(pair_feat_prev)

        # merge pair_feat_prev
        pair_feat = torch.cat((pair_feat, pair_feat_prev), dim=-1)


        # concat 1D feature
        feat_1D_pair = self.get_pair_feat_1D(feat_1D_left, feat_1D_right, seq_weight)
        pair_feat = torch.cat((feat_1D_pair, pair_feat), dim=-1)


        # 2D project down
        pair_feat = pair_feat.permute(0, 3, 1, 2).contiguous()
        pair_feat = self.proj_2D(pair_feat)

        # ResNet2D blocks
        for block in self.ResNet2D_blocks:
            pair_feat = block(pair_feat)

        return pair_feat.permute(0, 2, 3, 1)

class Pair_to_MSA(nn.Module): 
    '''
    Using pairwise feature based Attention to update MSA embedding.
    '''
    def __init__(self, in_dim, out_dim, n_emb_2D, n_head, scaling, seqlen_scaling=True, dropout=0.0, bias=False, ):
        super().__init__()
        self.n_head = n_head
        self.scaling = scaling
        self.seqlen_scaling = seqlen_scaling

        # pair attn block
        self.linear_layer = nn.Linear(n_emb_2D, n_head, bias=bias)
        # V
        self.to_v = nn.Linear(in_dim, out_dim, bias=bias)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, msa_emb, pair_feat, attn_mask=None, ):
        # x: [n_batch, seq_num, seq_len, n_head*n_emb]
        n_batch, seq_num, seq_len, n_head_x_n_emb = msa_emb.size()
        scaling = self.scaling / math.sqrt(seq_len) if self.seqlen_scaling else self.scaling

        pair_feat = pair_feat * scaling
        # [n_batch, seq_len, seq_len, n_head] -> [n_batch, n_head, seq_len, seq_len]
        pair_feat = self.linear_layer(pair_feat).permute(0, 3, 1, 2)

        # [n_batch, seq_num, seq_len, seq_len, n_head, emb]
        V = self.to_v(msa_emb).view(n_batch, seq_num, seq_len, self.n_head, -1)

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, None] * attn_mask[:, None, :]
            pair_feat.masked_fill_(attn_mask[:, None, :, :]==0, torch.finfo(pair_feat.dtype).min)

        # attn, [n_batch, n_head, seq_num, seq_len, seq_len]
        attn = pair_feat.softmax(-1)

        # dropout
        attn = self.dropout(attn)

        # out
        out = torch.einsum(f"nhij,nrjhd->nrihd", attn, V)
        out = out.contiguous().view(n_batch, seq_num, seq_len, -1)

        return out

class MHPair_to_MSA(nn.Module): 
    '''
    Multi-head Pair_to_MSA.
    '''
    def __init__(self, emb_dim=512, n_emb_2D=256, n_head=4, bias=False, dropout=0.0, attn_dropout=0.0,
                        seqlen_scaling=True, head_by_head=True, *args, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_dim // n_head
        self.head_by_head = head_by_head
        scaling = self.head_dim ** -0.5

        self.layer_norm_msa = nn.LayerNorm(emb_dim)
        self.layer_norm_pair = nn.LayerNorm(n_emb_2D)

        # calc each head one after another
        if self.head_by_head:
            self.heads = nn.ModuleList()
            for i in range(n_head):
                self_attn = Pair_to_MSA(emb_dim, self.head_dim, n_emb_2D, 1, scaling=scaling, seqlen_scaling=seqlen_scaling, dropout=attn_dropout, )
                self.heads.append(self_attn)
        # calc all heads simultaneously
        else:
            self.heads = Pair_to_MSA(emb_dim, emb_dim, n_emb_2D, n_head, scaling=scaling, seqlen_scaling=seqlen_scaling, dropout=attn_dropout, )

        self.to_out = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_feat, attn_mask=None, **kwargs):
        # [n_batch, seq_num, seq_len, n_emb]
        msa_emb = self.layer_norm_msa(x)

        # [n_batch, seq_len, seq_len, n_dim]
        pair_feat = (pair_feat + pair_feat.transpose(1, 2)) * 0.5
        pair_feat = self.layer_norm_pair(pair_feat)

        # calc each head one after another
        if self.head_by_head:
            out = []
            for i in range(self.n_head):
                head = self.heads[i]
                head_out = head(msa_emb, pair_feat, attn_mask)
                out.append(head_out)
            out = torch.cat(out, dim=-1)
        # calc all heads simultaneously
        else:
            out = self.heads(msa_emb, pair_feat, attn_mask)

        out = self.to_out(out)
        out = self.dropout(out)

        x = x + out

        return x

class MSAPairAttentionLayer(nn.Module): 
    '''
    The main Module of pairwise feature based attention on MSA feature.
    '''
    def __init__(self,
                    emb_dim,
                    n_emb_2D,
                    n_ResNet2D_block,
                    AxialTransformer_config,
                    include_ff=True, n_ff_hidden=None, ff_dropout=0.0, activation='elu', bias=False,
                    pre_layernorm=True,
                    *args, **kwargs
                ):
        super().__init__()
        self.include_ff = include_ff

        # ResNet2D Block
        self.MSA_to_Pair_layer = MSA_to_Pair(n_emb_2D=n_emb_2D, n_ResNet2D_block=n_ResNet2D_block ,  emb_dim=emb_dim)
        self.n_emb_2D = self.MSA_to_Pair_layer.get_out_channel()

        # Axial Transformer
        self.axial_transformer_layer = AxialTransformerLayer(**AxialTransformer_config, emb_dim=self.n_emb_2D,)

        # multi-head pair attn
        self.mh_pair_attn_layer = MHPair_to_MSA(emb_dim=emb_dim, n_emb_2D=self.n_emb_2D)
        # Feed Forward
        if n_ff_hidden is None: n_ff_hidden = emb_dim
        ff_layer = FeedForward(emb_dim, emb_dim, n_hidden=n_ff_hidden, activation=activation, dropout=ff_dropout, bias=bias)
        self.ff_layer = Residual(ff_layer, emb_dim, emb_dim, pre_layernorm=pre_layernorm)

    def get_pair_feat_dim(self, ):
        return self.n_emb_2D

    def forward(self, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None, attn=None, **kwargs):
        # x: [n_batch, seq_num, seq_len, emb_dim]

        # gen pair feature
        pair_feat = self.MSA_to_Pair_layer(x, seq_weight, seq_mask, pair_feat, attn)

        # refine pair feature using AxialTransformer layers
        pair_feat = self.axial_transformer_layer(pair_feat, res_mask)

        # pair attn
        x = self.mh_pair_attn_layer(x, pair_feat, res_mask)

        # feed forward
        x = self.ff_layer(x)


        return x, pair_feat

class HybridAttnLayer(nn.Module):
    '''
    Hybrid Attention layer: TransformerLayer and MSAPairAttentionLayer layer.
    '''
    def __init__(self, emb_dim, Transformer_config, MSAPairAttn_config,  *args, **kwargs):
        super().__init__()

        self.transformer_layer = TransformerLayer(**Transformer_config, emb_dim=emb_dim, )
        self.pair_attn_layer = MSAPairAttentionLayer(**MSAPairAttn_config, emb_dim=emb_dim, )

    def get_pair_feat_dim(self, ):
        return self.pair_attn_layer.get_pair_feat_dim()

    def wrapper(self, fn, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None, attn=None, *args, **kwargs):

        x = fn(x, seq_weight, res_mask, seq_mask, pair_feat, attn)
        return x

    def forward(self, x, seq_weight=None, res_mask=None, seq_mask=None, pair_feat=None, *args, **kwargs):
        x = self.wrapper(self.transformer_layer, x, seq_weight, res_mask, seq_mask)
        x = self.wrapper(self.pair_attn_layer, x, seq_weight, res_mask, seq_mask, pair_feat)

        return x

class HybridAttn(nn.Module):
    '''
    HybridAttn, basic Attn (transformer) and Pair Attn.

    Refs:

    '''
    def __init__(self,
            n_input: int, # the config of embedding
            n_inner: int, # the feat dim of inner layers.
            output_config: dict, # the config of output.
            Transformer_config: dict, # the config of Transformer on MSA.
            MSAPairAttn_config: dict, # the config of MSA update based on pair attn.
            n_layer: int=6, # the number of attention layers.
            sqrt_seq_weight: bool=False, # use the sqrt(seq_weight).
            *args, **kwargs):
        super().__init__()

        self.sqrt_seq_weight = sqrt_seq_weight

        # gate layer
        self.gate_layer = None
        if not n_input==n_inner:
            print(self.__class__.__name__, f"n_input ({n_input}) != n_inner ({n_inner}), add gate layer")
            self.gate_layer = nn.Linear(n_input, n_inner, bias=False)

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(HybridAttnLayer(n_inner, Transformer_config, MSAPairAttn_config))

        # output layer
        n_pair_feat = self.layers[-1].get_pair_feat_dim()
        self.out_layer = MSAOutputModule(n_inner, **output_config, n_pair_feat=n_pair_feat)


    def forward(self, x, seq_weight=None, MSA_encoding=None, res_mask=None, seq_mask=None, global_token=None, pair_feat=None, recycle_query=False, ESM=None, *args, **kwargs):
        '''
        x shape: [n_batch, seq_num, seq_len, n_embedding]
        '''
        # seq weight
        _seq_weight = torch.sqrt(seq_weight) if self.sqrt_seq_weight else seq_weight

        # x.shape: n_batch, seq_num, seq_len, n_emb

  
        # gate layer
        if self.gate_layer is not None:
            x = self.gate_layer(x)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            x, pair_feat = layer(x, _seq_weight, res_mask, seq_mask, pair_feat)

        # x: (n_batch, seq_num, seq_len, n_emb)
        # pair_feat: (n_batch, seq_len, seq_len, n_emb)

        x = (x, pair_feat)

        if recycle_query: query_feat = x[0][:, 0, :, :]

        if isinstance(x, tuple) and recycle_query:
            return x[0], x[1], query_feat
        else:
            return x
