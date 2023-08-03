#!/bin/sh
source /cluster/scratch/ggabriel/rel_pos/venv_new/bin/activate

echo "Starting Training..."
CUDA_VISIBLE_DEVICES=0 fairseq-train ../data-bin/iwslt17.de-en.bpe16k/     \
	--encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 --encoder-layers 6 --encoder-attention-heads 8 \
	--decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-layers 6 --decoder-attention-heads 8 \
	--max-epoch 100  --task translation  \
	--arch transformer_iwslt_de_en --share-decoder-input-output-embed  \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps '1e-8' \
	--clip-norm 1.0  --lr '3e-4' --lr-scheduler inverse_sqrt \
	--warmup-updates 8000 --warmup-init-lr '1e-07' --label-smoothing 0 \
	--criterion label_smoothed_cross_entropy  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 \
	--weight-decay 0.0001 \
	--save-dir ./checkpoints/transformer --max-tokens 4096 --eval-bleu \
	--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
	--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--no-epoch-checkpoints --patience 20 \
	--max-epoch 200 --no-token-positional-embeddings

