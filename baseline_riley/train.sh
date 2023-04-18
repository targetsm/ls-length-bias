#!/bin/sh

ROOT=/cluster/home/ggabriel/ls-length-bias/baseline_riley/
DATA_DIR=$ROOT/iwslt17.de-en.bpe16k

echo"Binarize Dataset"
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $DATA_DIR/train.bpe.de-en \
    --validpref $DATA_DIR/valid0.bpe.de-en,$DATA_DIR/valid1.bpe.de-en,$DATA_DIR/valid2.bpe.de-en,$DATA_DIR/valid3.bpe.de-en,$DATA_DIR/valid4.bpe.de-en,$DATA_DIR/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10



