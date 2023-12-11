# 📏 ls-length-bias

Repository for my semester project: Understanding the Effects of Label Smoothing on Length Biases in Machine Translation

## Setup

We train transformer models using the [Fairseq](https://github.com/facebookresearch/fairseq) toolkit.

### Data
For most experiments we use the German to English IWSLT 2017 TED task dataset (Cettolo et al., 2012).
The data can be downloaded from https://wit3.fbk.eu/2017-01-c.

Similar to  [Riley and Chiang, 2022](https://arxiv.org/abs/2210.10817) and [Ott et al., 2018](https://github.com/darcey/transformers_without_tears), we apply a copy filter before bpe to remove sentences that have a overlap of over 50% between source and target.


## Riley and Chiang baseline

We replicated the experiments on label smoothing of [Riley and Chiang, 2022](https://arxiv.org/abs/2210.10817).
Setup adapted from [fairseq/examples/translation](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt17-multilingual.sh).
Hyper-parameters partly taken from [transformers without tears](https://github.com/darcey/transformers_without_tears).
The results are shown in [Baseline Riley & Chiang.pdf](Baseline%20Riley%20%26%20Chiang.pdf)

### How to run the experiment:

1. Clone [Fairseq](https://github.com/facebookresearch/fairseq) and replace `fairseq/scripts/spm_encode.py` with `baseline_riley/scripts/spm_encode.py`.
2. Prepare the dataset using `baseline_riley/scripts/prepare-iwslt17.sh`
3. Start training with `baseline_riley/scripts/train.sh`
4. You can plot the results with the python scripts found in [baseline_riley/img](baseline_riley/img).

## Ngram experiments

We run experiments with ngrams, estimating ngrams on artificial and real data.
We further interpolate the estimated distribution with the uniform distribution using the label smoothing $\lambda$.

## Transformer experiments
Finally we recreated 
