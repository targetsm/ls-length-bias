#!/bin/sh

mkdir -p evaluation_nols
fairseq-generate data-bin/iwslt17.de-en.bpe16k --path checkpoints/transformer_nols/checkpoint_best.pt \
	--batch-size 128 --beam 4 --remove-bpe --sacrebleu --results-path evaluation_nols

