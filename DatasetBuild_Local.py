#!/usr/bin/env python3
# encoding: utf-8

import os
import pickle, bz2

def read_name_list(name):
    with open(name, 'r') as fh:
        content = [line.strip().split(':')[0] for line in list(fh)]
    return content

def build_dataset(name, A2M_format,
                ALN_format=None, TPL_format=None, TGT_format=None, TPL_list_format=None,
                targets=None):

    all_data = []
    data_targets = []
    for target in targets:
        MSA_file = A2M_format%(target)

        if os.path.isfile(MSA_file):
            MSA = [_.split()[0].upper() for _ in open(MSA_file).readlines()]
            seq_len, seq_num = len(MSA[0]), len(MSA)

            data = {
                'target': target,
                'A2M': MSA_file,
                'seq_len': seq_len,
                'seq_num': seq_num,
            }

            # Template and Alignment
            if ALN_format is not None and TPL_format is not None:
                TPL_files = []
                ALN_files = []
                TPL_files = read_name_list(TPL_list_format % target)
                for tplname in TPL_files:
                    ALN_files.append(ALN_format % (tplname, target))
                TPL_files = [TPL_format % tpl_name for tpl_name in TPL_files]
                data['Alignment'] = ALN_files
                data['Template'] = TPL_files
                # data['TargetFeat'] = TGT_format % (target, target, target)
                data['TargetFeat'] = TGT_format % (target)


            all_data.append(data)
            data_targets.append(target)

    print('Not used targets:', set(targets)-set(data_targets))
    pkl_file = "%s.pkl.bz2"%(name)
    print(pkl_file, len(all_data))
    # set_trace()
    if len(all_data)>0:
        pickle.dump(all_data, bz2.BZ2File(pkl_file, "wb"))

msa = './data/msa/%s.a2m'
aln_format='./data/aln/%s-%s.fasta'
tpl_format='./data/template/%s.tpl.pkl'
tpl_list_format='./data/templatelist/%s.templatelist'
tgt_list_format='./data/tgt/%s.tgt'

# Build example dataset
build_dataset('example',
             msa,
             ALN_format=aln_format, TPL_format=tpl_format,
             TPL_list_format=tpl_list_format,
             TGT_format=tgt_list_format,
             targets=['T1084-D1'])
