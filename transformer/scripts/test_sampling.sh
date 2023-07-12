#!/bin/sh

DIR=/cluster/scratch/ggabriel/transformer/base

jid=$(sbatch -o test_sampling_0.001.log --gpus=1 --gres=gpumem:24g --mem-per-cpu=8000 --time=24:00:00 --wrap="fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/ls_0.001/checkpoints/transformer/checkpoint_best.pt --unnormalized --max-len-b 300 --batch-size 1 --beam 1000 --nbest 1000 --sampling --sacrebleu --results-path $DIR/ls_0.001/evaluation/sampling")
for i in 0.005 0.01 0.05 0.1; 
do
	mkdir -p $DIR/ls_$i/evaluation/sampling
	echo	$jid
	jid=$(sbatch -o test_sampling_$i.log --gpus=1 --gres=gpumem:24g --mem-per-cpu=8000 --time=24:00:00 --depend=afterany:$jid --wrap="fairseq-generate $DIR/data-bin/iwslt17.de-en.bpe16k  --path $DIR/ls_$i/checkpoints/transformer/checkpoint_best.pt --unnormalized --max-len-b 300 --batch-size 1 --beam 1000 --nbest 1000 --sampling --sacrebleu --results-path $DIR/ls_$i/evaluation/sampling")
done
