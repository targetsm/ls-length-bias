import sys
import matplotlib.pyplot as plt
import math

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'base':[], 'short':[], 'long':[]}
bleu_dict = {'base':[], 'short':[], 'long':[]}
ref_dict = {'base':[], 'short':[], 'long':[]}
se_dict = {'base':[], 'short':[], 'long':[]}

for d in hyp_dict.keys():
    for l in ls_list:
        f = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/no_pos/evaluation/ls_{l}/generate-test-sampling.txt','r')
        i = 0
        hyp_list = []
        ref_len = 0
        hyp_len = 0
        hyp_se = 0
        total_count = 0
        src_len = 0
        for line in f:
            #print(line)
            #print(eval(line))
            if d == 'short':
                if i > 8079//4:
                    break
                else:
                    i += 1
            if d == 'long':
                if i < 8079//4:
                    i+= 1
                    continue
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
plt.errorbar(ls_list, hyp_dict['base'], se_dict['base'], marker='o', linestyle='dashed', label='base', color='blue', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], label='base reference', color='blue')
training_avg = 0
training_file = open('../data/iwslt17.de-en.bpe16k/train.bpe.de-en.en').readlines()
for line in training_file:
    training_avg += len(line.split())
training_avg /= len(training_file)
plt.hlines(training_avg, ls_list[0], ls_list[-1], label='train reference', color='red')
plt.errorbar(ls_list, hyp_dict['short'], se_dict['short'], marker='o', linestyle='dashed', label='short', color='orange', capsize=4)
plt.hlines(ref_dict['short'][0], ls_list[0], ls_list[-1], label='short reference', color='orange')
plt.errorbar(ls_list, hyp_dict['long'], se_dict['long'], marker='o', linestyle='dashed', label='long', color='green', capsize=4)
plt.hlines(ref_dict['long'][0], ls_list[0], ls_list[-1], label='long reference', color='green')

plt.grid()
plt.title('Sampling average sentence length for length splits')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.legend()
plt.savefig('img/split.pdf', dpi=400, bbox_inches='tight')
plt.clf()
