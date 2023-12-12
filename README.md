# 📏 ls-length-bias

Repository for my semester project: Understanding the Effects of Label Smoothing on Length Biases in Machine Translation

## Setup

### Requirements
- [Fairseq](https://github.com/facebookresearch/fairseq) toolkit v0.12.3
- [sacreBLEU](https://github.com/mjpost/sacrebleu)

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
4. Sample from the trained fairseq model using ...TODO
5. You can plot the results with the python scripts found in [baseline_riley/img](baseline_riley/img).

## Ngram experiments

We run experiments on label smoothing applied to ngrams using artificial and real data.
To apply label smoothing to the ngram model we interpolate the estimated distribution with the uniform distribution using the label smoothing $\lambda$.
The results of our experiments are shown in [Ngram experiments](Ngram%20experiments.pdf).

Code for model generation and sampling can be found in [ngram/del.py](ngram/del.py).
To generate a model run:
```
python -u del.py -n 3 --task generate --dict_path data-bin/iwslt17.de-en.bpe16k/dict.txt --data_path iwslt17.de-en.bpe16k/test.bpe.de-en.en --model_path /cluster/scratch/ggabriel/ngram/model_3gram_16k --ls_eps 0.1
```
To sample from the generated model, run:
```
python -u del.py -n 3 --task sample --dict_path data-bin/iwslt17.de-en.bpe16k/dict.txt --data_path iwslt17.de-en.bpe16k/test.bpe.de-en.en --model_path /cluster/scratch/ggabriel/ngram/model_3gram_16k --ls_eps 0.1 --output_path sample_3gram_16k_norep/ls_0.1.txt
```

### Data

For experiments with real data, we used the same script as before [prepare-iwslt17.sh](baseline_riley/scripts/prepare-iwslt17.sh) without truncation. We fitted ngrams only to the english part of the training data.

The artificial data used in our experiments can be found at [data](ngram/small/data). The script used to generate the data can be found at [generate_text.py](ngram/small/generate_text.py).

## Transformer experiments

We train transformers with different configurations and varying label smoothing parameter.
Code for training, sampling and plotting can be found under [transformer](transformer).
The results of our experiments are shown in [Transformer experiments](Transformer%20experiments.pdf).

### Relative positional embeddings
Fairseq v0.10.2

