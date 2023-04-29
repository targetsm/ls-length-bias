

beams="1 2 3 4 5 10 20"
sent="10 20 30 40 50 60 70 80 90 100"

for s in $sent; do
	for b in $beams; do
		path="$HOME/ls-length-bias/baseline_riley/s_$s/evaluation/beams/$b/generate-test.txt"
		grep ^H $path | cut -f3- > gen.out.sys 
		grep ^T $path | cut -f2- > gen.out.ref

		echo "$s, $b"
		python lr.py gen.out.sys gen.out.ref

		rm gen.out.sys
		rm gen.out.ref
	done

done
