import sys
import matplotlib.pyplot as plt
import math

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'base':[], '4k':[], '64k':[]}
bleu_dict = {'base':[], '4k':[], '64k':[]}
ref_dict = {'base':[], '4k':[], '64k':[]}
se_dict = {'base':[], '4k':[], '64k':[]}
std_dict = {'base':[], '4k':[], '64k':[]}

for d in hyp_dict.keys():
    for l in ls_list:
        f = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/{d}/evaluation/ls_{l}/generate-test-sampling.txt','r')
        i = 1
        hyp_list = []
        ref_len = 0
        hyp_len = 0
        hyp_se = 0
        hyp_std = 0
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
            hyp_se += math.sqrt(sum([(x - hyp_sum/1000)**2 for x in ls[2:]])/1000)/math.sqrt(1000)* 2.576
            hyp_std += math.sqrt(sum([(x - hyp_sum/1000)**2 for x in ls[2:]])/1000)
            total_count += 1
        hyp_dict[d].append(hyp_len/total_count)
        se_dict[d].append(hyp_se / total_count)
        std_dict[d].append(hyp_std / total_count)
        ref_dict[d].append(ref_len / total_count)
        print(hyp_dict[d], se_dict[d], ref_dict[d])
plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(3.9*1.2,1.7*1.2))
plt.errorbar(ls_list, hyp_dict['4k'], se_dict['4k'], marker='o', label='4k', color='orange', capsize=4)
plt.hlines(ref_dict['4k'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='4k ref', color='orange')
plt.errorbar(ls_list, hyp_dict['base'], se_dict['base'], marker='o', label='16k', color='blue', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='16k ref', color='blue')
plt.errorbar(ls_list, hyp_dict['64k'], se_dict['64k'], marker='o', label='64k', color='green', capsize=4)
plt.hlines(ref_dict['64k'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='64k ref', color='green')
plt.grid()
#plt.title('Sampling mean sentence length for different bpe dictionary sizes')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/sampling.pdf', dpi=400)
plt.clf()

plt.errorbar(ls_list, hyp_dict['4k'], std_dict['4k'], marker='o', label='4k', color='orange', capsize=4)
plt.hlines(ref_dict['4k'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='4k ref', color='orange')
plt.errorbar(ls_list, hyp_dict['base'], std_dict['base'], marker='o', label='16k', color='blue', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='16k ref', color='blue')
plt.errorbar(ls_list, hyp_dict['64k'], std_dict['64k'], marker='o', label='64k', color='green', capsize=4)
plt.hlines(ref_dict['64k'][0], ls_list[0], ls_list[-1], linestyle='dashed', label='64k ref', color='green')
plt.grid()
#plt.title('Sampling mean sentence length for different bpe dictionary sizes')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/sampling_std.pdf', dpi=400)
plt.clf()

