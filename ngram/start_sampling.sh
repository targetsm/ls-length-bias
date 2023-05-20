

for i in 0 0.001 0.01 0.1 0.25 0.5 1 
do
        sbatch -o sampling_3gram_4k.log --mem-per-cpu=512G --time=24:00:00 --wrap="python -u del.py -n 3 --task sample --dict_path data-bin/iwslt17.de-en.bpe4k/dict.txt --data_path iwslt17.de-en.bpe4k/test.bpe.de-en.en --model_path /cluster/scratch/ggabriel/ngram/model_3gram_4k --ls_eps $i --output_path sample_3gram_4k_norep/ls_$i.txt"
done
for i in 0 0.001 0.01 0.1 0.25 0.5 1 
do
	sbatch -o sampling_5gram_4k.log --mem-per-cpu=512G --time=24:00:00 --wrap="python -u del.py -n 5 --task sample --dict_path data-bin/iwslt17.de-en.bpe4k/dict.txt --data_path iwslt17.de-en.bpe4k/test.bpe.de-en.en --model_path /cluster/scratch/ggabriel/ngram/model_5gram_4k --ls_eps $i --output_path sample_5gram_4k_norep/ls_$i.txt"
done
for i in 0 0.001 0.01 0.1 0.25 0.5 1
do
        sbatch -o sampling_3gram_16k.log --mem-per-cpu=512G --time=24:00:00 --wrap="python -u del.py -n 3 --task sample --dict_path data-bin/iwslt17.de-en.bpe16k/dict.txt --data_path iwslt17.de-en.bpe16k/test.bpe.de-en.en --model_path /cluster/scratch/ggabriel/ngram/model_3gram_16k --ls_eps $i --output_path sample_3gram_16k_norep/ls_$i.txt"
done
