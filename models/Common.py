#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn
import math

from IPython.core.debugger import set_trace

def activation_func(activation, inplace=False):
    '''
    Activation functions
    '''
    if activation is None: return None
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['elu', nn.ELU(inplace=inplace)],
        ['leaky_relu', nn.LeakyReLU(inplace=inplace)],
        ['selu', nn.SELU(inplace=inplace)],
        ['none', nn.Identity()],
    ])[activation]

def normalization_func(input_size, normalization, n_dim):
    '''
    Normalization functions
    '''
    assert input_size in ['1D', '2D'], 'input_size: 1D or 2D.'
    if input_size=='1D':
        return  nn.ModuleDict([
            ['batch', nn.BatchNorm1d(n_dim)],
            ['instance', nn.InstanceNorm1d(n_dim)],
            ['layer', nn.LayerNorm(n_dim)],
            ['none', nn.Identity()],
        ])[normalization]

    elif input_size=='2D':
        return  nn.ModuleDict([
            ['batch', nn.BatchNorm2d(n_dim)],
            ['instance', nn.InstanceNorm2d(n_dim)],
            ['none', nn.Identity()]
        ])[normalization]

class FeedForward(nn.Module):
    '''
    Feed Forward Layer.
    '''
    def __init__(self, n_input, n_output, n_hidden=None, activation="elu", dropout=0.0, bias=False):
        super().__init__()
        self.n_hidden = n_hidden
        w1_out = n_output if n_hidden is None else n_hidden
        self.w_1 = nn.Linear(n_input, w1_out, bias=bias)
        self.activation = activation_func(activation)
        self.dropout = nn.Dropout(dropout)
        if n_hidden is not None:
            self.w_2 = nn.Linear(n_hidden, n_output, bias=bias)

    def forward(self, x, *args, **kwargs):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.n_hidden is not None:
            x = self.w_2(x)
        return x

class Residual(nn.Module): 
    '''
    Residual wrapper.
    '''
    def __init__(self, fn, n_input, n_output, pre_layernorm=True, dropout=0.0, bias=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(n_input) if pre_layernorm else nn.LayerNorm(n_output)
        self.dropout = nn.Dropout(dropout)
        self.pre_layernorm = pre_layernorm

    def forward(self, x, *args, **kwargs):
        residual = x

        # pre layernorm
        if self.pre_layernorm: x = self.norm(x)

        outputs = self.fn(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        # post layernorm
        if not self.pre_layernorm: x = self.norm(x)

        # dropout
        x = self.dropout(x)

        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x

class SinCosPositionalEncoding(nn.Module):
    '''
    Sin and Cos Positional Encoding (1D).
    '''
    def __init__(self,
                    n_emb: int, # the embedding dim.
                ):
        super().__init__()
        self.n_emb = n_emb
        inv_freq = 1. / (10000 ** (torch.arange(0, n_emb, 2).float() / n_emb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, idxs, ):
        '''
        idxs: (n_batch, seq_len)
        return: (n_batch, 1, seq_len, n_emb)
        '''
        idxs = idxs.type_as(self.inv_freq)
        sin_inp_x = torch.einsum("bi,j->bij", idxs, self.inv_freq)
        emb = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        return emb[:, None, ]

class RelativePositionalEncoding2D(nn.Module): 
    '''
    Relative Positional Encoding.
    '''
    def __init__(self,
                n_emb: int, # the dim of embedding
                max_gap: int=32, # the max gap
                *args, **kwargs
            ):
        super().__init__()
        self.max_gap = max_gap
        self.n_index = max_gap * 2 + 1
        self.pos_emb = nn.Linear(self.n_index, n_emb)

    def forward(self, idxs, ):
        '''
        idxs: (n_batch, seq_len)
        return: (n_batch, seq_len, seq_len, n_emb)
        '''
        pos_2d = (idxs[:, None, :] - idxs[:, :, None])
        pos_2d = pos_2d.clip(min=-self.max_gap, max=self.max_gap) + self.max_gap

        pos_2d_onehot = torch.eye(self.n_index)[pos_2d.long(), :].type_as(self.pos_emb.weight)

        return self.pos_emb(pos_2d_onehot)

class EmbeddingModule(nn.Module): 
    '''
    EmbeddingModule: generate the input feature.
    '''
    def __init__(self,
            n_alphabet: int, # the alphabet size of residue index, n_alphabet+1 is padding value
            n_emb_MSA: int, # the embedding size of MSA feat
            # query
            feat_msa_transformer: bool=False, # use embedding from MSATransformer
            msa_transformer_config: dict={},  # config of MSAtransformer
            # pair
            n_emb_pair: int=288, # the embedding size of pair feat
            max_gap: int=32, # max gap for 2D Positional Embedding (Relative)

            *args, **kwargs):
        super().__init__()
        self.n_alphabet = n_alphabet
        self.feat_msa_transformer = feat_msa_transformer # True


        n_MSA_query_emb = n_emb_MSA

        assert n_MSA_query_emb % 2 == 0
        n_MSA_query_emb = int(n_MSA_query_emb/2)

        # MSA emb
        # self.MSA_emb_layer = nn.Linear(n_MSA_input, n_MSA_query_emb)
        self.MSA_emb_layer = nn.Embedding(n_alphabet+1, n_MSA_query_emb, padding_idx=n_alphabet) # n_alphabet+1 is padding value

        # query emb
        self.query_emb_layer = nn.Embedding(n_alphabet+1, n_MSA_query_emb, padding_idx=n_alphabet)

        # MSA positional emb
        # PosEmb_MSA_type SinCos
        self.PosEmb_MSA_layer = SinCosPositionalEncoding(n_emb_MSA, )

        # MSATransformer emb
        if feat_msa_transformer:
            import esm
            self.MSA_MODEL = esm.pretrained.load_model_and_alphabet_core(
                torch.load(msa_transformer_config['model_path'],
                           map_location='cpu'),
                None)[0]
            # froze the parameters
            for name, param in self.MSA_MODEL.named_parameters():
                param.requires_grad = False
            self.msa_transformer_emb_linear = nn.Linear(
                msa_transformer_config['n_embedding'], n_emb_MSA)

        # pair emb
        self.left_linear = nn.Embedding(n_alphabet+1, n_emb_pair, padding_idx=n_alphabet)
        self.right_linear = nn.Embedding(n_alphabet+1, n_emb_pair, padding_idx=n_alphabet)

        # pair positional emb
        # PosEmb_pair_type Relative
        self.PosEmb_pair_layer = RelativePositionalEncoding2D(n_emb_pair, max_gap=max_gap, )

        self.query_recycle_norm = nn.LayerNorm(n_emb_MSA)
        self.query_recycle_linear = nn.Linear(n_emb_MSA*2, n_emb_MSA)

        self.pair_recycle_norm = nn.LayerNorm(n_emb_pair)
        self.pair_recycle_linear = nn.Linear(n_emb_pair*2, n_emb_pair)

    def forward(self, MSA_encoding, seq_encoding, res_idxs, cluster_profile=None,
                MSA_token=None, recycle_data=None, *args, **kwargs):
        n_batch, seq_num, seq_len = MSA_encoding.shape[:3]

        # MSA_emb
        MSA_emb = self.MSA_emb_layer(MSA_encoding)

        # MSA_emb: merge query emb
        query_emb = self.query_emb_layer(seq_encoding)
        query_emb = query_emb[:, None, :, :].repeat(1, seq_num, 1, 1)
        MSA_emb = torch.cat([query_emb, MSA_emb], dim=-1) # (n_batch, seq_num, seq_len, n_emb)

        # MSA_emb: merge cluster profile emb

        # MSA_emb: add positional emb
        MSA_emb = MSA_emb + self.PosEmb_MSA_layer(res_idxs).type_as(MSA_emb)

        # MSATransformer: add msa transformer emb
        if self.feat_msa_transformer:
            MSAtransformer_emb =  self.generate_msa_transformer_emb(
                MSA_token, seq_num, seq_len)
            MSA_emb = MSA_emb + self.msa_transformer_emb_linear(MSAtransformer_emb)

        # pair emb
        x_left = self.left_linear(seq_encoding)
        x_right = self.right_linear(seq_encoding)
        pair_emb = x_left[:, :, None] + x_right[:, None]

        pair_emb = pair_emb + self.PosEmb_pair_layer(res_idxs).type_as(pair_emb)

        # add recycle feat
        prev_query_feat = recycle_data['query_feat'] if recycle_data is not None else torch.zeros_like(MSA_emb[:, 0, :, :])
        prev_query_feat = self.query_recycle_norm(prev_query_feat)
        MSA_emb[:, 0, :, :] = self.query_recycle_linear(torch.cat((MSA_emb[:, 0, :, :], prev_query_feat), dim=-1))

        prev_pair_feat = recycle_data['pair_feat'] if recycle_data is not None else torch.zeros_like(pair_emb)
        prev_pair_feat = self.pair_recycle_norm(prev_pair_feat)
        pair_emb = self.get_pair_emb(pair_emb, prev_pair_feat)

        return MSA_emb, pair_emb

    def get_pair_emb(self, pair_emb, prev_pair_feat=None):

        pair_feat = torch.cat((pair_emb, prev_pair_feat), dim=-1)
        pair_emb = self.pair_recycle_linear(pair_feat)

        return pair_emb

    def generate_msa_transformer_emb(self, MSA_token, seq_num, seq_len):
        MSA_emb = self.MSA_MODEL(
            MSA_token, repr_layers=[12],
            return_contacts=False)['representations'][12][:, :, 1:seq_len+1]
        return MSA_emb

class MSAOutputModule(nn.Module):
    '''
    MSAOutputModule: generate the out feature of MSA_module.
    '''
    def __init__(self,
            n_input: int, # emb size of input tensor (1D)
            n_output_1D: int, # emb size of output tensor (1D)
            n_inner_emb: int, # emb size of inner tensor (1D) for 2D feature
            n_output_2D: int, # emb size of output tensor (2D)
            activation: str='elu', # activation function.
            normalization: str='instance', # normalization function.
            bias: bool=False,
            n_pair_feat: int=None, # the pair_feat channel if use pair_feat
            *args, **kwargs):
        super().__init__()
        self.n_input = n_input
        self.n_output_1D = n_output_1D
        self.n_inner_emb = n_inner_emb
        self.n_output_2D = n_output_2D


        self.emb_fc_2D = nn.Sequential(
            nn.Linear(n_input, n_inner_emb, bias=False),
            normalization_func('1D', 'layer', n_inner_emb),
            activation_func(activation),
        )

        n_inner_2D = self.n_inner_emb * 2 + n_pair_feat

        self.proj_layer_2D = None
        if n_inner_2D != n_output_2D:
            from .ResNet2D import Conv2dAuto
            self.proj_layer_2D = nn.Sequential(
                Conv2dAuto(n_inner_2D, n_output_2D, kernel_size=1, dilation=1, bias=bias),
                normalization_func('2D', normalization, n_output_2D),
                activation_func(activation),
            )


    def forward(self, x, seq_weight=None, MSA_encoding=None, seq_mask=None, ESM=None, *args, **kwargs):
        if torch.is_tensor(x): MSA_feat = x
        elif isinstance(x, tuple): MSA_feat = x[0]

        pair_feat = x[1]

        n_batch, seq_num, seq_len, n_input = MSA_feat.shape
        grid = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len))

        x_query = MSA_feat[:, 0, :, :] # (n_batch, 1, seq_len, n_input)
        emb = self.emb_fc_2D(x_query)
        # (n_batch, seq_len, seq_len, n_emb*2)
        emb = torch.cat([emb[:, grid[0], :], emb[:, grid[1], :]], dim=-1)

        feat_2D = torch.cat([emb, pair_feat], dim=-1)
        # (n_batch, seq_len, seq_len, n_emb) -> (n_batch, n_emb, seq_len, seq_len)
        feat_2D = feat_2D.permute(0, 3, 1, 2)
        feat_2D = self.proj_layer_2D(feat_2D)

        return feat_2D

