import sys
import matplotlib.pyplot as plt
import math

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'base':[], 'pos_learned':[], 'no_pos':[]}
bleu_dict = {'base':[], 'pos_learned':[], 'no_pos':[]}
ref_dict = {'base':[], 'pos_learned':[], 'no_pos':[]}
se_dict = {'base':[], 'pos_learned':[], 'no_pos':[]}

for d in hyp_dict.keys():
    for l in ls_list:
        f = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/{d}/evaluation/ls_{l}/generate-test-sampling.txt','r')
        i = 1
        hyp_list = []
        ref_len = 0
        hyp_len = 0
        hyp_se = 0
        total_count = 0
        src_len = 0
        for line in f:
            #print(line)
            #print(eval(line))
            ls = eval(line)
            src_len += ls[0]
            ref_len += ls[1]
            hyp_sum = sum(ls[2:])
            hyp_len += hyp_sum/1000
            hyp_se += math.sqrt(sum([(x - hyp_sum/1000)**2 for x in ls[2:]])/1000)/math.sqrt(1000)
            total_count += 1
        hyp_dict[d].append(hyp_len/total_count)
        se_dict[d].append(hyp_se / total_count)
        ref_dict[d].append(ref_len / total_count)
        print(hyp_dict[d], se_dict[d], ref_dict[d])
plt.figure(figsize=(12,7))
plt.errorbar(ls_list, hyp_dict['base'], se_dict['base'], marker='o', linestyle='dashed', label='16k (base)', color='blue', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], label='reference', color='blue')
plt.errorbar(ls_list, hyp_dict['pos_learned'], se_dict['pos_learned'], marker='o', linestyle='dashed', label='learned', color='orange', capsize=4)
plt.errorbar(ls_list, hyp_dict['no_pos'], se_dict['no_pos'], marker='o', linestyle='dashed', label='no_pos', color='green', capsize=4)
plt.grid()
plt.title('Sampling average sentence length for different positional embeddings')
plt.ylabel('length')
plt.xlabel('label smoothing alpha')
plt.legend()
plt.savefig('img/pos_sampling.png', dpi=400)
plt.clf()
