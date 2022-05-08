#!/usr/bin/env python3
# encoding: utf-8

from IPython.core.debugger import set_trace
import torch
from torch import nn

from .Common import EmbeddingModule

class MainModel(nn.Module):
    '''
    MainModel: implement all modules.
    '''
    def __init__(self, configs):
        super().__init__()

        # embedding module
        self.emb_module = EmbeddingModule(**configs['Embedding_module'])

        # MSA module
        from .HybridAttn import HybridAttn
        self.MSA_module = HybridAttn(**configs['MSA_module'])

        # Template module
        from .TemplateModel import TemplateAxialTransformer
        self.template_emb = TemplateAxialTransformer(**configs['Template_module'])

        # pair module
        from .PairModule import PairModule
        self.pair_module = PairModule(**configs['Pair_module'])


        self.configs = configs
        self.n_alphabet = configs['FEATURE']['n_alphabet']

    def forward(self, data, *args, **kwargs):
        seq_encoding = data['seq_encoding'] # (n_batch, seq_len)
        MSA_encoding = data['MSA_encoding'] # (n_batch, seq_num, seq_len)
        seq_weight = data['seq_weight'] # (n_batch, seq_num)
        res_idxs = data['res_idxs'] # (n_batch, seq_len)
        ESM = data['ESM'] # (n_batch, seq_len, C)
        MSA_token = data['MSA_token'] # (n_batch, seq_num, seq_len)
        cluster_profile = data['cluster_profile'] # (n_batch, seq_num, seq_len, n_alphabet)
        recycling = data['recycling']
        recycle_data = data['recycle_data']

        n_batch, seq_num, seq_len = MSA_encoding.shape

        # residue mask, (n_batch, seq_len)
        res_mask = torch.where(seq_encoding < self.n_alphabet, 1., 0.)

        # sequence mask
        seq_mask = torch.where(seq_weight>0, 1., 0.)

        # emb layer
        MSA_feat, pair_feat = self.emb_module(MSA_encoding, seq_encoding, res_idxs, cluster_profile, MSA_token, recycle_data)

        # global token
        global_token = None
        # self.add_global_token: False

        # data for next recycle
        new_recycle_data = {}
        recycle_query = torch.tensor(1) if ('query' in self.configs['RECYCLING']['recycle_feats'] and recycling) else torch.tensor(0)
        recycle_pair = torch.tensor(1) if ('pair' in self.configs['RECYCLING']['recycle_feats'] and recycling) else torch.tensor(0)

        # template module
        template_feat1d = data['template_feat1d']
        template_feat2d = data['template_feat2d']
        template_mask = data['template_mask']
        x_query = MSA_feat[:, 0, :, :]     # (n_batch, seq_len, n_input)

        pair_feat = self.template_emb(x_query, pair_feat, template_feat1d, template_feat2d, res_mask, template_mask)

        # MSA module
        x = self.MSA_module(MSA_feat, seq_weight, MSA_encoding, res_mask, seq_mask, global_token, pair_feat, recycle_query)

        # recycle query feat
        if recycle_query: x, new_recycle_data['query_feat'] = x[:-1], x[-1].detach()

        # MSA_MODULE out layer
        # MSA_MODULE out layer: True
        feat_2D = self.MSA_module.out_layer(x, seq_weight, MSA_encoding, seq_mask, ESM, )

        outputs = ()

        # pair module
        # pair module: True
        pair_module_out, feat_2D = self.pair_module(feat_2D, res_mask, )

        feat_2D = feat_2D.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]

        # recycle pair feat
        if recycle_pair:
            new_recycle_data['pair_feat'] = feat_2D.detach()

        return pair_module_out, new_recycle_data
