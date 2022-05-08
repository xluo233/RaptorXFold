#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn

from .ResNet2D import ResNet2D

from IPython.core.debugger import set_trace

class OutputLayer(nn.Module):
    def __init__(self, n_input, pred_config,  *args, **kwargs):
        super().__init__()

        # output layers
        self.symmetric_before_out, self.symmetric_final_out = [], []
        self.output_layers = nn.ModuleDict({})
        for pred_name in pred_config:
            _n_output = pred_config[pred_name]['n_output']
            self.output_layers.update({pred_name: ResNet2D(n_input, _n_output*2, 1, _n_output)})
            if pred_config[pred_name]['symmetric_before_out_layer']: self.symmetric_before_out.append(pred_name)
            if pred_config[pred_name]['symmetric_final_out']: self.symmetric_final_out.append(pred_name)

    def forward(self, x):
        # symmetric before out layer
        if len(self.symmetric_before_out)>0: x_symmetric = (x + x.transpose(2, 3)) * 0.5

        outputs = ()
        for pred_name in self.output_layers:
            # symmetric before out layer
            if pred_name in self.symmetric_before_out:
                out = self.output_layers[pred_name](x_symmetric)
            else:
                out = self.output_layers[pred_name](x)

            # symmetric final out
            if pred_name in self.symmetric_final_out:
                out = (out + out.transpose(2, 3)) * 0.5

            outputs += (out, )
        
        return outputs

class PairModule(nn.Module):
    '''
    PairModule for inter-residue distance and orientation prediction.
        Args:
            n_input (int): input channel size.
            n_hidden (int): hidden channel size.
    '''
    def __init__(self, model_config, pred_config,  *args, **kwargs):
        super().__init__()


        self.pair_module = ResNet2D(**model_config)

        self.n_pair_channel = self.pair_module.get_output() 
        self.output_layer = OutputLayer(self.n_pair_channel, pred_config)
        
    def get_pair_feat_channel(self, ):
        return self.n_pair_channel

    def forward(self, x, res_mask=None, *args, **kwargs):
        '''
        Input should be (B, C, H, W)
        '''

        x = self.pair_module(x, res_mask)
        x = x.contiguous() # https://github.com/pytorch/pytorch/issues/48439
        outputs = self.output_layer(x)
        
        return outputs, x
