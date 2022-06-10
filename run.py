from doctest import Example
import os, sys, socket
import pickle, bz2
from argparse import ArgumentParser
import torch

os.environ['Attn4ProStrucPred_Dir'] = os.getcwd()

from models.MainModel import MainModel
from dataset.MSADataset import MSADataset, _gen_minibatches, mBatchSampler
from torch.utils.data import DataLoader
from configs import CONFIGS
from utils import Utils

def main(args):

    # global seed
    global_seed = 0 
    CONFIGS['global_seed'] = global_seed

    # MSATransformer parameter
    CONFIGS['FEATURE']['msa_transformer_config']['model_path'] = args.msa_transformer_model_path
    CONFIGS['Embedding_module']['msa_transformer_config'] = CONFIGS['FEATURE']['msa_transformer_config']

    # load data
    dataset_type = args.dataset_type
    data, minibatches = _gen_minibatches(CONFIGS['DATASET'][dataset_type])
    dataset = MSADataset(data, CONFIGS, dataset_type, torch.device('cpu'))
    batch_sampler = mBatchSampler(minibatches, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset._minibatch_collate_fn, num_workers=0)

    # load model
    checkpoint = torch.load(args.resume_from_checkpoint)
    model = MainModel(CONFIGS)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # prediction
    pred_names = [*CONFIGS['PREDICTION']]

    for cycle_batch in dataloader:
        if not os.path.isdir(CONFIGS['DATASET'][dataset_type]['out_dir']): os.makedirs(CONFIGS['DATASET'][dataset_type]['out_dir'])
        target = cycle_batch[0][0]['target'][0]
        out_file = f"{CONFIGS['DATASET'][dataset_type]['out_dir']}/{target}.pkl.bz2"
        print(f"Output path {out_file}")

        n_cycle = len(cycle_batch[0])
        recycle_data = None
        for i in range(n_cycle):
            x = cycle_batch[0][i]
            
            x['recycling'] = i<n_cycle-1
            x['recycle_data'] = recycle_data

            preds, recycle_data = model.forward(x)

        # save prediction
        prediction = {}

        prediction['preds'] = {}
        for i, pred_name in enumerate(pred_names):
            if not CONFIGS['PREDICTION'][pred_name]['save']: continue
            pred = preds[i][0]
            if len(pred.shape) == 3 and (pred.shape[-1] == pred.shape[-2]):
                pred = Utils.pair_pred_transform(pred)
            prediction['preds'][pred_name] = pred.cpu().detach().numpy()
        pickle.dump(prediction, bz2.BZ2File(out_file, "wb"))

if __name__ == '__main__':

    # Argument Parser
    parser = ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--msa_transformer_model_path', type=str, default='./utils/esm_msa1b_t12_100M_UR50S.pt', help='MSATransformer parameters')
    parser.add_argument('--dataset_type', type=str, default='test')
    args = parser.parse_args()

    main(args)
