

for filename in data/*.txt; do
	base=$(basename "$filename" .txt)
	j=3	
		python del.py -t generate -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base.model
		for i in 0 0.001 0.005 0.01 0.05 0.1; do
			python del.py -t sample -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base.model --output_path samples/$base\_$i.txt --ls_eps $i
		done
done

base='8-10-1000'
python del.py -t generate -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model
for i in 0 0.001 0.005 0.01 0.05 0.1; do
        python del.py -t sample -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model --output_path samples/$base\_$i\_$j\gram_v2.txt --ls_eps $i
done

python del.py -t generate -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model
for i in 0 0.001 0.005 0.01 0.05 0.1; do
        python del.py -t sample -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model --output_path samples/$base\_$i\_$j\gram_v3.txt --ls_eps $i
done

for j in 4 5 8 10; do
	python del.py -t generate -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model
	for i in 0 0.001 0.005 0.01 0.05 0.1; do
		python del.py -t sample -n $j --data_path data/$base.txt --dict_path data/$base.dict --model_path models/$base\_$j\gram.model --output_path samples/$base\_$i\_$j\gram.txt --ls_eps $i
	done
done


