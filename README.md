# 📏 ls-length-bias

Repository for my semester project: Understanding the Effects of Label Smoothing on Length Biases in Machine Translation

## Data
For most experiments we use the IWSLT2017 DE-EN translation task data.
The data can be downloaded from https://wit3.fbk.eu/2017-01-c.

Similar to  [Riley and Chiang, 2022](https://arxiv.org/abs/2210.10817), we apply a copy filter before bpe to remove sentences that have a overlap of over 50% between source and target [Ott et al., 2018](https://github.com/darcey/transformers_without_tears).

## Riley and Chiang baseline

We replicated the experiments on label smoothing of [Riley and Chiang, 2022](https://arxiv.org/abs/2210.10817).
Setup adapted from [fairseq/examples/translation](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt17-multilingual.sh).
Hyper-parameters partly taken from [transformers without tears](https://github.com/darcey/transformers_without_tears).

### How to run the experiment:

1. Clone [Fairseq](https://github.com/facebookresearch/fairseq) and replace `fairseq/scripts/spm_encode.py` with `baseline_riley/scripts/spm_encode.py`.
2. Prepare the dataset using `baseline_riley/scripts/prepare-iwslt17.sh`
3. Start training with `baseline_riley/scripts/train.sh`
4. You can plot the results with the python scripts found in [baseline_riley/img](baseline_riley/img).

## Ngram experiments
We then proceeded to experiments with ngrams in ...
Results can be seen here.

## Transformer experiments
Finally we recreated 
