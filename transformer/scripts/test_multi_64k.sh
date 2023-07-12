#!/bin/sh

for model in 4k 64k; do

DIR=/cluster/scratch/ggabriel/transformer/$model

mkdir -p ../$model/evaluation
fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.001/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.001 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.005/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.005 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.01/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.01 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.05/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.05 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.1/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.1 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe$model  --path $DIR/ls_0.1/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path ../$model/evaluation/ls_0.5 --num-workers 2
