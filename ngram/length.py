import sys

ref = sys.argv[1]
gen = sys.argv[2]

len_ref = 0
total_len_ref = 0
total_len_gen = 0
with open(ref) as r:
    for line in r:
        len_ref += len(line.split())
        total_len_ref += 1
len_gen = 0
with open(gen) as g:
    for line in g:
        len_gen += len(line.split())
        total_len_gen += 1
print(len_ref, len_gen, len_ref/total_len_ref, len_gen/total_len_gen, len_gen/len_ref,  (- len_ref/total_len_ref) + len_gen/total_len_gen)
