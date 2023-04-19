#!/bin/sh

ROOT=/cluster/home/ggabriel/ls-length-bias/baseline_riley/
DATA_DIR=$ROOT/iwslt17.de-en.bpe16k

echo "Binarize Dataset"
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $DATA_DIR/train.bpe.de-en \
    --validpref $DATA_DIR/valid0.bpe.de-en,$DATA_DIR/valid1.bpe.de-en,$DATA_DIR/valid2.bpe.de-en,$DATA_DIR/valid3.bpe.de-en,$DATA_DIR/valid4.bpe.de-en,$DATA_DIR/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de-en.bpe16k \
    --workers 10

echo "Starting Training..."
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de-en.bpe16k/     \
	--max-epoch 100 --ddp-backend=legacy_ddp  --task translation  \
	--arch transformer_iwslt_de_en --share-decoder-input-output-embed  \
	--optimizer adam --adam-betas '(0.9, 0.98)'  \
	--clip-norm 0.0  --lr 0.0005 --lr-scheduler inverse_sqrt \
	--warmup-updates 4000 --warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--criterion label_smoothed_cross_entropy  --dropout 0.3 --weight-decay 0.0001 \
	--save-dir checkpoints/transformer --max-tokens 4000 --eval-bleu \
	--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
	--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--ignore-unused-valid-subsets --no-epoch-checkpoints --patience 20

