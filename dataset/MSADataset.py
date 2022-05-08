#!/usr/bin/env python3
# encoding: utf-8

import os
import pickle
import bz2
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from operator import itemgetter

from utils import Utils
from IPython.core.debugger import set_trace

def _item_filter(item, config):
    if item['seq_len'] < config['min_seq_len']:
        return False

    if ('resolution' in config) and ('resolution' in item):
        if item['resolution'] < config['resolution'][0]:
            return False

        if item['resolution'] > config['resolution'][1]:
            return False

    return True

def _data_filter(data, config):
    filtered_data = []
    for item in data:
        if isinstance(item, list):
            filtered_items = [_ for _ in item if _item_filter(_, config)]
            if len(filtered_items)>0: filtered_data.append(filtered_items)
        elif isinstance(item, dict):
            if _item_filter(item, config):
                filtered_data.append(item)

    return filtered_data


def _build_minibatches(data, idxs, config):
    # build minibatches
    minibatches = []
    minibatch, batch_max_seq_num = [], 0
    for i in idxs:
        item = data[i]
        batch_max_seq_num = max(batch_max_seq_num, min(config['max_seq_num'], item['seq_num']))
        batch_max_seq_len = min(config['max_seq_len'], item['seq_len'])
        n_token = (len(minibatch)+1) * batch_max_seq_len * batch_max_seq_num
        n_2D_token = (len(minibatch)+1) * batch_max_seq_len**2
        if n_token<config['max_tokens'] and n_2D_token<config['max_seq_len']**2 and len(minibatch)+1<config['max_batch_size']:
            minibatch.append(i)
        else:
            if len(minibatch)>0: minibatches.append(minibatch)
            minibatch, batch_max_seq_num = [i, ], min(config['max_seq_num'], item['seq_num'])

    if len(minibatch)>0: minibatches.append(minibatch) # the last minibatch

    if config['shuffle']: random.shuffle(minibatches)

    batch_sizes = [len(_) for _ in minibatches] + [0,]
    print(f"\t sample: {sum(batch_sizes)}, minibatch: {len(minibatches)}, max batch: {max(batch_sizes)}")

    return minibatches

def _gen_minibatches(config):
    # load data
    data = pickle.load(bz2.BZ2File(config['dataset_info'], "rb"))

    # filter
    data = _data_filter(data, config)

    # do not sample seqs
    if not config['pre_sample_seqs']:
        pass
    # seq_idxs exists, just use it
    elif isinstance(data[0], dict) and ('seq_idxs' in data[0]):
        print("use sampled seq_idxs")
    # sample seqs from MSA
    else:
        print("sample seq_idxs")
        assert isinstance(data[0], dict), "sample seq_idxs not support cluster dataset"
        new_data = []
        for item in data:
            # sample and shuffle idxs, if seq_num==0, only use the query seq
            _idxs = np.arange(1, item['seq_num']).tolist()
            _seq_num = min(int((item['seq_num']-1) * config['MSA_sample_rate']), config['max_seq_num'])
            _seq_idxs = [0, ] + random.sample(_idxs, k=_seq_num)
            _item = item.copy()
            _item['seq_idxs'] = _seq_idxs
            new_data.append(_item)

        data = new_data

    data = sorted(data, key=lambda i: i['seq_len'])
    idxs = np.arange(0, len(data))
    minibatches = _build_minibatches(data, idxs, config)
    
    return data, minibatches

class mBatchSampler():
    def __init__(self, minibatches, shuffle=False, batch_size=1, drop_last=True):
        assert batch_size==1, "only support batch_size=1 so far"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.minibatches = minibatches

    def __len__(self, ):
        return len(self.minibatches)

    def __iter__(self, ):
        if self.shuffle: random.shuffle(self.minibatches)

        for indexs in self.minibatches:
            yield indexs


class MSADataset(Dataset):
    def __init__(self, data, configs, dataset_type, device):
        """
            data: the samples.
            configs: configurations.
            dataset_type: train, val, or test.
        """
        super().__init__()

        self.configs = configs

        self.pin_memory = configs['DATASET']['pin_memory']
        self.padding_idx = configs['FEATURE']['n_alphabet'] 
        self.device = device

        # dataset
        self.data = data
        self.dataset_type = dataset_type
        dataset_config = configs['DATASET'][dataset_type]
        self.A2M_folder = dataset_config['A2M_folder']
        self.GroundTruth_folder = dataset_config['GroundTruth_folder']
        self.max_seq_num = dataset_config['max_seq_num']
        self.max_seq_len = dataset_config['max_seq_len']
        self.max_tokens = dataset_config['max_tokens']
        self.use_fixed_MSA = dataset_config['use_fixed_MSA']
        self.random_segment = dataset_config['random_segment']
        self.continuous_segment = dataset_config['continuous_segment']
        self.MSA_sample_rate = dataset_config['MSA_sample_rate']

        # feature
        self.n_alphabet = configs['FEATURE']['n_alphabet']
        self.calc_seq_weight = configs['FEATURE']['calc_seq_weight']
        self.seq_sim_cutoff = configs['FEATURE']['seq_sim_cutoff']
        self.concat_query_embedding = configs['FEATURE']['concat_query_embedding']
        self.add_relative_position = configs['FEATURE']['add_relative_position']
        self.feat_cluster_profile = configs['FEATURE']['feat_cluster_profile']
        self.MLM_config = configs['FEATURE']['MLM_config']
        self.max_full_MSA_seq_num = int(configs['FEATURE']['max_full_MSA_seq_num'])
        self.feat_msa_transformer = configs['FEATURE']['feat_msa_transformer']
        if self.feat_msa_transformer:
            import esm
            alphabet = esm.pretrained.load_model_and_alphabet_core(
                torch.load(configs['FEATURE']['msa_transformer_config']['model_path'],
                           map_location='cpu'), None)[1]
            self.msa_batch_converter = alphabet.get_batch_converter()

        # template config
        self.ALN_folder = dataset_config['ALN_folder']
        self.TPL_folder = dataset_config['TPL_folder']
        self.TGT_folder = dataset_config['TGT_folder']

        self.max_templates = dataset_config['max_templates']
        self.random_template_sample = dataset_config['random_template_sample']

        self.template_feat = configs['FEATURE']['template_feat']
        self.n_template_feat_1d = configs['FEATURE']['n_template_feat_1d']
        self.n_template_feat_2d = configs['FEATURE']['n_template_feat_2d']

        # label
        self.labels = configs['LABEL']

        # recycle
        self.n_max_cycle = configs['RECYCLING']['n_max_cycle']


    def __getitem__(self, idx):
        # print('idx:', idx)
        if torch.is_tensor(idx): idx = idx.tolist()

        samples = []
        n_cycle = random.randint(1, self.n_max_cycle) if self.dataset_type=='train' else self.n_max_cycle
        self.cycle_sampled_res_idx = {}
        for _ in range(n_cycle):
            if isinstance(idx, list):
                _samples = [self.get_one_sample(i) for i in idx]
            else:
                _samples = [self.get_one_sample(idx), ]
            samples.append(_samples)

        return samples

    def get_valid_res_idxs(self, res_missing, max_gap=100, min_segment=6):
        '''
        get residue index of valid segments.
        '''
        res_mask = np.ma.MaskedArray(res_missing, mask=res_missing)

        # merge segments separated by short gap segment
        for _slice in np.ma.clump_masked(res_mask):
            if len(res_mask[_slice]) <= max_gap:
                res_mask[_slice] = False

        # del short segment
        for _slice in np.ma.clump_unmasked(res_mask):
            if len(res_mask[_slice]) < min_segment:
                res_mask[_slice] = True

        # get valid_res_idxs
        valid_res_idxs = np.arange(len(res_missing))[res_mask == False]

        return valid_res_idxs

    def sample_segment(self, res_missing, max_seq_len, random_seq_fragment=False, continuous_segment=True):
        '''
        sample segment based on res_missing.
        '''
        valid_res_idxs = self.get_valid_res_idxs(res_missing)
        n_valid_res = len(valid_res_idxs)

        if n_valid_res <= max_seq_len: # whole seq
            res_idxs = valid_res_idxs

        else:
            # sample the central segment
            if not random_seq_fragment:
                seq_i = int((n_valid_res-max_seq_len)/2)
                seq_j = seq_i + max_seq_len
                res_idxs = valid_res_idxs[seq_i:seq_j]

            # random sample continuous segment
            elif continuous_segment:
                seq_i = np.random.randint(n_valid_res - max_seq_len)
                seq_j = seq_i + max_seq_len
                res_idxs = valid_res_idxs[seq_i:seq_j]

            # random sample two discontinuous segments (equal length)
            else:
                segment_len = int(max_seq_len/2)

                seq_i1 = np.random.randint(0, n_valid_res-segment_len+1)
                seq_j1 = seq_i1 + segment_len
                res_idxs1 = valid_res_idxs[seq_i1:seq_j1]

                candidate_i2 = np.concatenate((np.arange(0, seq_i1-segment_len), np.arange(seq_j1, n_valid_res-segment_len+1)), axis=0)
                seq_i2 = np.random.choice(candidate_i2, 1)[0]
                seq_j2 = seq_i2 + segment_len
                res_idxs2 = valid_res_idxs[seq_i2:seq_j2]

                if seq_i1 < seq_i2:
                    res_idxs = np.concatenate((res_idxs1, res_idxs2), axis=0)
                else:
                    res_idxs = np.concatenate((res_idxs2, res_idxs1), axis=0)

        return res_idxs

    def get_one_sample(self, idx):
        # print(self.dataset_type, self.data[idx])
        item_info = self.data[idx]
        target = item_info['target']

        # MSA
        A2M_file = self.A2M_folder+item_info['A2M'] if (self.dataset_type in ['train', 'val']) else item_info['A2M']
        with open(A2M_file) as fr: full_MSA = [seq.strip().upper() for seq in fr.readlines()][:self.max_full_MSA_seq_num]
        query_seq, seq_len = full_MSA[0], len(full_MSA[0])

        # sample residue index
        if target in self.cycle_sampled_res_idx:
            res_idxs = self.cycle_sampled_res_idx[target]
        else:
            res_missing = np.zeros(seq_len)

            res_idxs = self.sample_segment(res_missing, self.max_seq_len, self.random_segment, self.continuous_segment)
            self.cycle_sampled_res_idx[target] = res_idxs

        res_num = len(res_idxs)

        n_full_MSA_seq = len(full_MSA)
        if n_full_MSA_seq > 1: # sample MSA
            if 'seq_idxs' in item_info: # use pre sampled seq_idxs
                _seq_num = min(int(self.max_tokens/res_num), self.max_seq_num)
                seq_idxs = item_info['seq_idxs'][:_seq_num]
            elif self.use_fixed_MSA:
                _seq_num = min(int(n_full_MSA_seq*self.MSA_sample_rate), self.max_seq_num, int(self.max_tokens/res_num))
                seq_idxs = list(range(_seq_num))
            else: # sample sequence on-the-fly
                _seq_num = min(int((n_full_MSA_seq-1)*self.MSA_sample_rate), self.max_seq_num-1, int(self.max_tokens/res_num-1))
                seq_idxs = [0, ] + random.sample(np.arange(1, n_full_MSA_seq).tolist(), k=_seq_num)
            MSA = itemgetter(*seq_idxs)(full_MSA)
            if isinstance(MSA, str): MSA = [MSA, ]
        else:
            seq_idxs = [0, ]
            MSA = full_MSA

        seq_encoding = Utils.MSA_encoding([query_seq, ], n_alphabet=self.n_alphabet, )[0][res_idxs]
        assert max(seq_encoding)<21, f"Unknown residue type in query sequence: {query_seq}."

        assert len(MSA) >= 1, "No seq in MSA."
        MSA_encoding = Utils.MSA_encoding(MSA, n_alphabet=self.n_alphabet, )[:, res_idxs]

        if self.calc_seq_weight:
            seq_weight = Utils.calc_seq_weight(MSA_encoding, n_alphabet=self.n_alphabet, sim_cutoff=self.seq_sim_cutoff)
        else:
            seq_weight = np.ones((len(MSA_encoding), ))
        MSA_encoding = Utils.concat_query_embedding(MSA_encoding[0], MSA_encoding) if self.concat_query_embedding else MSA_encoding
        if self.add_relative_position: MSA_encoding = Utils.add_relative_position(MSA_encoding)

        feature = {
            'seq_encoding': seq_encoding.astype(np.int),
            'MSA_encoding': MSA_encoding.astype(np.int),
            'seq_weight': seq_weight.astype(np.float32),
            'res_idxs': res_idxs.astype(np.int),
            'cluster_profile':  None,
            'extra_MSA_encoding': None, 
            'extra_seq_weight': None, 
        }

        if self.feat_msa_transformer:
            raw_MSA = [('', msa_item) for msa_item in MSA]
            _, _, token = self.msa_batch_converter(raw_MSA)
            feature['MSA_token'] = torch.cat(
                (token[0][:, [0]], token[0][:, res_idxs+1]), -1)

        # template feature
        TPL_files = self.data[idx]['Template']
        ALN_files = self.data[idx]['Alignment']

        if len(TPL_files) > 0:
            # template sampling, Uniform (1, max_templates)
            if self.random_template_sample:
                TPL_sample_num = random.randint(1, min(self.max_templates, len(TPL_files)))
                tpl_idxs = random.sample(np.arange(0, len(TPL_files)).tolist(), k=TPL_sample_num)
            else:
                tpl_idxs = np.arange(0, min(self.max_templates, len(TPL_files)))

            TPL_files = itemgetter(*tpl_idxs)(TPL_files)
            ALN_files = itemgetter(*tpl_idxs)(ALN_files)
            if not isinstance(TPL_files, tuple):
                TPL_files = (TPL_files, )
            if not isinstance(ALN_files, tuple):
                ALN_files = (ALN_files, )

            if self.dataset_type in ['train', 'val']:
                TPL_files = [self.TPL_folder + tpl_file for tpl_file in TPL_files]
                ALN_files = [self.ALN_folder + aln_file for aln_file in ALN_files]
                TGT_file = self.TGT_folder + self.data[idx]['TargetFeat']
            else:
                TGT_file = self.data[idx]['TargetFeat']

            template_feat1d = Utils.TemplateFeature1d(TGT_file, TPL_files, ALN_files, self.template_feat['feat1d'])[:, res_idxs]
            template_feat2d = Utils.TemplateFeature2d(TGT_file, TPL_files, ALN_files, self.template_feat['feat2d'])[:, res_idxs][:, :, res_idxs]
        else:
            template_feat1d = np.zeros((1, res_num, self.n_template_feat_1d))
            template_feat2d = np.zeros((1, res_num, res_num, self.n_template_feat_2d))

        feature['template_feat1d'] = template_feat1d.astype(np.float32)
        feature['template_feat2d'] = template_feat2d.astype(np.float32)

        return [feature]

    def _minibatch_collate_fn(self, cycle_samples):

        for i, _ in enumerate(cycle_samples): cycle_samples[i] = cycle_samples[i][0]

        # n_cycle = len(cycle_samples)
        samples0 = cycle_samples[0]
        batch_size = len(samples0)

        seq_nums, seq_lens, extra_seq_nums = [], [], []
        tpl_nums = []
        for item in samples0:
            seq_nums.append(item[0]['MSA_encoding'].shape[0])
            seq_lens.append(item[0]['MSA_encoding'].shape[1])

            tpl_nums.append(item[0]['template_feat1d'].shape[0])

        max_seq_num, max_seq_len, min_seq_len = max(seq_nums + [2, ]), max(seq_lens), min(seq_lens)
        max_extra_seq_num = max(extra_seq_nums + [1, ])

        # template feature, not sample in recycle iteration
        max_templ = max(tpl_nums)
        batch_template_feat1d = torch.zeros(
            (batch_size, max_templ, max_seq_len, self.configs['FEATURE']['n_template_feat_1d']),
            dtype=torch.float32, device=self.device)

        batch_template_feat2d = torch.zeros(
            (batch_size, max_templ, max_seq_len, max_seq_len, self.configs['FEATURE']['n_template_feat_2d']),
            dtype=torch.float32, device=self.device)

        batch_template_mask = torch.zeros(
            (batch_size, max_templ),
            dtype=torch.float32, device=self.device)

        for i, item in enumerate(samples0):
            feature = item[0]
            MSA_encoding = torch.as_tensor(feature['MSA_encoding'], dtype=torch.int64, device=self.device)
            seq_num, seq_len = MSA_encoding.shape

            template_feat1d = torch.as_tensor(feature['template_feat1d'], dtype=torch.float32, device=self.device)
            template_feat2d = torch.as_tensor(feature['template_feat2d'], dtype=torch.float32, device=self.device)
            num_templ = template_feat1d.shape[0]

            batch_template_feat1d[i, :num_templ, :seq_len] = template_feat1d
            batch_template_feat2d[i, :num_templ, :seq_len, :seq_len] = template_feat2d
            batch_template_mask[i, :num_templ] = 1

        cycle_batch_feature = []
        for _c, samples in enumerate(cycle_samples):
            # create tensor
            # feature, labels, sample_info
            batch_feature = {
                'seq_encoding': torch.zeros((batch_size, max_seq_len), dtype=torch.int64, device=self.device) + self.padding_idx,
                'MSA_encoding': torch.zeros((batch_size, max_seq_num, max_seq_len), dtype=torch.int64, device=self.device) + self.padding_idx,
                'seq_weight': torch.zeros((batch_size, max_seq_num), dtype=torch.float32, device=self.device),
                'res_idxs': torch.zeros((batch_size, max_seq_len), dtype=torch.int16, device=self.device) - 1,

                'ESM': None,
                'cluster_profile': torch.zeros((batch_size, max_seq_num, max_seq_len, self.configs['FEATURE']['n_alphabet']), dtype=torch.float32, device=self.device) if self.configs['FEATURE']['feat_cluster_profile'] else None,
                'extra_MSA_encoding': None, 
                'extra_seq_weight':None, 

                'seq_onehot': None,
                'PSSM': None,
                'DCA':  None,

                'template_feat1d': batch_template_feat1d ,
                'template_feat2d': batch_template_feat2d,
                'template_mask': batch_template_mask,
                'MSA_token': torch.ones((batch_size, max_seq_num, max_seq_len+1), dtype=torch.int64, device=self.device) if self.configs['FEATURE']['feat_msa_transformer'] else None
            }


            # fill the tensor
            for i, item in enumerate(samples):
                # feature
                feature = item[0]
                seq_encoding = torch.as_tensor(feature['seq_encoding'], dtype=torch.int64, device=self.device)
                MSA_encoding = torch.as_tensor(feature['MSA_encoding'], dtype=torch.int64, device=self.device)
                seq_weight = torch.as_tensor(feature['seq_weight'], dtype=torch.float32, device=self.device)
                res_idxs = torch.as_tensor(feature['res_idxs'], dtype=torch.int16, device=self.device)

                seq_num, seq_len = MSA_encoding.shape

                batch_feature['seq_encoding'][i, :seq_len] = seq_encoding
                batch_feature['MSA_encoding'][i, :seq_num, :seq_len] = MSA_encoding
                batch_feature['seq_weight'][i, :seq_num] = seq_weight
                batch_feature['res_idxs'][i, :seq_len] = res_idxs

                if self.configs['FEATURE']['feat_msa_transformer']:
                    batch_feature['MSA_token'][i, :seq_num, :seq_len+1] = torch.as_tensor(feature['MSA_token'], dtype=torch.int64, device=self.device)


            # print(f"B:{batch_size}, N:{max_seq_num}, L:{max_seq_len}")

            if self.pin_memory:
                [batch_feature[_key].pin_memory() for _key in batch_feature if torch.is_tensor(batch_feature[_key])]
            cycle_batch_feature.append(batch_feature)

        return [cycle_batch_feature]
