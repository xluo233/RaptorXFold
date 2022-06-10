# RaptorXFold

## Environment

## conda env
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
conda create --name RaptorXFold python=3.6 -y
conda activate RaptorXFold

```

## required package
* pytorch

    `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`


* numpy_groupies

    `pip install numpy_groupies`
* iminuit

    `pip install 'iminuit<2'`
* biopython

    `pip install biopython`
* ipython
    
    `pip install ipython`

* scipy

    `pip install scipy`

* [fair-esm](https://github.com/facebookresearch/esm)

    `pip install fair-esm`

## Download esm pt
```
wget -P ./utils  https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt
```
## Running example

Example Data
```
tar -xjf data.tar.bz2
```

Prepare data
```
python DatasetBuild_Local.py
```

Run predict
```
python3 run.py  --resume_from_checkpoint=./checkpoints/RaptorXFold.pt
```

Output
```
./Predictions/{target}.pkl.bz2
```

## Prediction
Output save the predicted distance bin for `CbCb`, `Ca1Cb1Cb2Ca2`, `N1Ca1Cb1Cb2` and `Ca1Cb1Cb2`
```
prediction : {
    pred : {
        CbCb,
        Ca1Cb1Cb2Ca2,
        N1Ca1Cb1Cb2,
        Ca1Cb1Cb2 
    }
}
```

## Prepare Data
Prepare data for `DatasetBuild_Local`, details are explianed in section **Materials and methods**.

- MSA: [HHblits](https://github.com/soedinglab/hh-suite), [DeepMSA](https://zhanggroup.org/DeepMSA/)

- Template: [NDThreader](https://github.com/wufandi/DL4SequenceAlignment)