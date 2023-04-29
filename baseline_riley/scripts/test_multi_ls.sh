#!/bin/sh

DIR=/cluster/scratch/ggabriel/baseline_riley/s_10

mkdir -p evaluation/beams
fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 1 --remove-bpe --sacrebleu --results-path evaluation/beams/ls1 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 2 --remove-bpe --sacrebleu --results-path evaluation/beams/ls2 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 3 --remove-bpe --sacrebleu --results-path evaluation/beams/ls3 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path evaluation/beams/ls4 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 5 --remove-bpe --sacrebleu --results-path evaluation/beams/ls5 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 128 --beam 10 --remove-bpe --sacrebleu --results-path evaluation/beams/ls10 --num-workers 2

fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/checkpoints/transformer/checkpoint_best.pt \
        --unnormalized --max-len-b 300 --batch-size 32 --beam 20 --remove-bpe --sacrebleu --results-path evaluation/beams/ls20 --num-workers 2
