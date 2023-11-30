# Riley and Chiang baseline

This is a replication of the experiments on label smoothing of [Riley and Chiang, 2022](https://arxiv.org/abs/2210.10817).
Setup adapted from [fairseq/examples/translation](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt17-multilingual.sh).
Hyper-parameters partly taken from [transformers without tears](https://github.com/darcey/transformers_without_tears).

## Data & Preprocessing

The data can be downloaded from https://wit3.fbk.eu/2017-01-c

As in the paper, we apply a copy filter before bpe to remove sentences that have a overlap of over 50% between source and target. [Ott et al., 2018](https://github.com/darcey/transformers_without_tears)

## How to run the experiment:

1. Clone [Fairseq](https://github.com/facebookresearch/fairseq) and replace `fairseq/scripts/spm_encode.py` with `baseline_riley/scripts/spm_encode.py`.
2. Prepare the dataset using `bash prepare-iwslt17.sh`
3. Start training with `bash train.sh`
4. You can plot the results with the python scripts found in [test](img) 
