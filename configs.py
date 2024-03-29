


CONFIGS = {
    'RECYCLING':{
        'n_max_cycle': 4,
        'recycle_feats': ['query', 'pair'],
    },
    'Embedding_module':{
            'n_alphabet': 23, 
            'n_emb_MSA': 384,
            'n_emb_pair': 288,
            'feat_msa_transformer': True,
        },
    'MSA_module': {
            'n_input': 384,
            'n_inner': 384,
            'n_layer': 8,
            'sqrt_seq_weight': False,
            'Transformer_config' : {
                'n_head':6,
                'out_dropout': 0.0,
                'attn_dropout': 0.0,
                'seqlen_scaling': True,
                'n_ff_hidden': 1536,
                'activation': 'elu',
                'ff_dropout': 0.0,
                'pre_layernorm': True,
                'head_by_head': True,
            },
            'MSAPairAttn_config' : {
                'include_ff': True,
                'n_ff_hidden': 1536,
                'n_emb_2D': 288, 
                'n_ResNet2D_block': 1,
                'AxialTransformer_config': {
                    'n_head': 8,
                    'n_ff_hidden': 1152, 
                    'head_by_head': True,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'activation': 'elu',
                }
            },
            'output_config': {
                'n_output_1D': 384,
                'n_inner_emb': 48,
                'n_output_2D': 96,
                'activation':'elu',
            }
        },
    'Template_module': {
            'n_t1d': 11,
            'n_t2d': 129,
            'n_q1d': 384,
            'n_q2d': 288,
            'n_inner': 64, 
            'n_output': 128,
            'n_layer': 4,

            # axial transformer
            'n_head': 4,
            'n_ff_hidden': 256, 
            'head_by_head': True,
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'activation': 'relu',
        },
    'Pair_module': {
            'model_config': {
                'n_input': 96,
                'n_output': 288,
                'n_channel': 96,
                'n_block': 72,
                'kernel_size': 3,
                'dilation': [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8],
                'dropout': 0.0,
                'activation': 'elu',
                'normalization': 'instance',
                'bias': False,
                'checkpoint_level': 'C1',
                'checkpoint_step': 1,
            },
            'pred_config' : {'CbCb': {'type': 'distance',
                'symmetric_before_out_layer': False,
                'symmetric_final_out': True,
                'save': True,
                'n_output': 37},
                'Ca1Cb1Cb2Ca2': {'type': 'angle',
                'symmetric_before_out_layer': False,
                'symmetric_final_out': True,
                'save': True,
                'n_output': 25},
                'N1Ca1Cb1Cb2': {'type': 'angle',
                'symmetric_before_out_layer': False,
                'symmetric_final_out': False,
                'save': True,
                'n_output': 25},
                'Ca1Cb1Cb2': {'type': 'angle',
                'symmetric_before_out_layer': False,
                'symmetric_final_out': False,
                'save': True,
                'n_output': 13}}
        },
    'FEATURE': {'calc_seq_weight': True,
                'seq_sim_cutoff': 0.8,
                'feat_cluster_profile': False,
                'feat_seq_onehot': False,
                'feat_DCA': False,
                'feat_PSSM': False,
                'n_ESM': 1280,
                'concat_query_embedding': False,
                'add_relative_position': False,
                'n_alphabet': 23,
                'max_full_MSA_seq_num': 1000000.0,
                'max_extra_MSA_seq_num': 1024,
                'MLM_config': {'replace_fraction': 0.15,
                'probs': {'uniform_prob': 0.1, 'profile_prob': 0.1, 'same_prob': 0.1}},
                'template_feat': {'feat1d': ['SeqSimlarity',
                'ProfileSimlarity',
                'GapPosition'],
                'feat2d': ['CaCa_disc_value',
                'CbCb_disc_value',
                'NO_disc_value',
                'Ca1Cb1Cb2_sincos',
                'N1Ca1Cb1Cb2_sincos',
                'Ca1Cb1Cb2Ca2_sincos']},
                'msa_transformer_config': {'n_embedding': 768,
                'model_path': None},
                'feat_msa_transformer': True,
                'n_template_feat_1d': 11,
                'n_template_feat_2d': 129,
                'feat_ESM': False
    },
    'LABEL':{'CbCb': {'type': 'distance',
                    'invalid_entry_separated': False,
                    'bins': [ 0. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ,
                            7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5, 11. , 11.5, 12. , 12.5,
                            13. , 13.5, 14. , 14.5, 15. , 15.5, 16. , 16.5, 17. , 17.5, 18. ,
                            18.5, 19. , 19.5, 20. ]
                    },
            'Ca1Cb1Cb2Ca2': {'type': 'angle',
                            'dist_cutoff': 20.0,
                            'invalid_entry_separated': False,
                            'bins': [-180., -165., -150., -135., -120., -105.,  -90.,  -75.,  -60.,
                                    -45.,  -30.,  -15.,    0.,   15.,   30.,   45.,   60.,   75.,
                                    90.,  105.,  120.,  135.,  150.,  165.,  180.]
                            },
            'N1Ca1Cb1Cb2': {'type': 'angle',
                            'dist_cutoff': 20.0,
                            'invalid_entry_separated': False,
                            'bins': [-180., -165., -150., -135., -120., -105.,  -90.,  -75.,  -60.,
                                    -45.,  -30.,  -15.,    0.,   15.,   30.,   45.,   60.,   75.,
                                    90.,  105.,  120.,  135.,  150.,  165.,  180.]
                            },
            'Ca1Cb1Cb2':    {'type': 'angle',
                            'dist_cutoff': 20.0,
                            'invalid_entry_separated': False,
                            'bins': [  0.,  15.,  30.,  45.,  60.,  75.,  90., 105., 120., 135., 150.,
                                    165., 180.]
                            },
         },

    'DATASET':{
        'pin_memory': False,

        'test':{'dataset_name': 'Example',
                'A2M_folder': '',
                'ALN_folder': '',
                'TGT_folder': '',
                'TPL_folder': '',
                'shuffle': False,
                'pre_sample_seqs': False,
                'use_fixed_MSA': True,
                'MSA_sample_rate': 1.0,
                'max_seq_num': 1024,
                'max_templates': 5,
                'random_template_sample': False,
                'last_min_seq_num': 1000,
                'max_batch_size': 1,
                'min_seq_len': 0,
                'max_seq_len': 2000,
                'max_tokens': 4000000.0,
                'symmetric_map': True,
                'random_segment': False,
                'continuous_segment': True,
                'dataset_info': './example.pkl.bz2',
                'out_dir': './Predictions'}
    },

    'PREDICTION':{
        'CbCb': {  'type': 'distance',
                    'symmetric_before_out_layer': False,
                    'symmetric_final_out': True,
                    'save': True,
                    'n_output': 37},
        'Ca1Cb1Cb2Ca2': {'type': 'angle',
                    'symmetric_before_out_layer': False,
                    'symmetric_final_out': True,
                    'save': True,
                    'n_output': 25},
        'N1Ca1Cb1Cb2': {'type': 'angle',
                    'symmetric_before_out_layer': False,
                    'symmetric_final_out': False,
                    'save': True,
                    'n_output': 25},
        'Ca1Cb1Cb2': {'type': 'angle',
                    'symmetric_before_out_layer': False,
                    'symmetric_final_out': False,
                    'save': True,
                    'n_output': 13},
    },
}
