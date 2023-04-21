#!/bin/sh

mkdir -p evaluation
fairseq-generate data-bin/iwslt17.de-en.bpe16k --path checkpoints/transformer/checkpoint_best.pt \
	--batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path evaluation

