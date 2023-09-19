import sys
import matplotlib.pyplot as plt
import math

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'4k':[], 'pos_learned_4k':[], 'no_pos_4k':[], 'rel_pos_4k':[]}
bleu_dict = {'4k':[], 'pos_learned_4k':[], 'no_pos_4k':[], 'rel_pos_4k':[]}
ref_dict = {'4k':[], 'pos_learned_4k':[], 'no_pos_4k':[], 'rel_pos_4k':[]}
se_dict = {'4k':[], 'pos_learned_4k':[], 'no_pos_4k':[], 'rel_pos_4k':[]}
std_dict = {'4k':[], 'pos_learned_4k':[], 'no_pos_4k':[], 'rel_pos_4k':[]}

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
            length = len(ls)-2
            src_len += ls[0]
            ref_len += ls[1]
            hyp_sum = sum(ls[2:])
            hyp_len += hyp_sum/length
            hyp_se += math.sqrt(sum([(x - hyp_sum/length)**2 for x in ls[2:]])/length)/math.sqrt(length)
            hyp_std += math.sqrt(sum([(x - hyp_sum/length)**2 for x in ls[2:]])/length)
            total_count += 1
        print(length)
        hyp_dict[d].append(hyp_len/total_count)
        se_dict[d].append(hyp_se / total_count)
        std_dict[d].append(hyp_std / total_count)
        ref_dict[d].append(ref_len / total_count)
        print(hyp_dict[d], se_dict[d], ref_dict[d])
plt.figure(figsize=(12,7))
plt.errorbar(ls_list, hyp_dict['4k'], se_dict['4k'], marker='o', label='sinusoid (4k)', color='blue', capsize=4)
plt.hlines(ref_dict['4k'][0], ls_list[0], ls_list[-1], label='reference', color='grey', linestyle='dashed')
plt.errorbar(ls_list, hyp_dict['pos_learned_4k'], se_dict['pos_learned_4k'], marker='o', label='learned', color='orange', capsize=4)
plt.errorbar(ls_list, hyp_dict['no_pos_4k'], se_dict['no_pos_4k'], marker='o', label='no_pos_4k', color='green', capsize=4)
plt.errorbar(ls_list, hyp_dict['rel_pos_4k'], se_dict['rel_pos_4k'], marker='o', label='rel_pos_4k', color='red', capsize=4)


training_avg = 0
training_file = open('../data/iwslt17.de-en.bpe4k/train.bpe.de-en.en').readlines()
for line in training_file:
    training_avg += len(line.split())
training_avg /= len(training_file)
plt.hlines(training_avg, ls_list[0], ls_list[-1], label='training set', color='grey', linestyle='dotted')

plt.grid()
plt.title('Sampling average sentence length for different positional embeddings (with SE)')
plt.ylabel('length')
plt.xlabel('label smoothing alpha')
plt.legend()
plt.savefig('img/pos_sampling_4k.png', dpi=400)
plt.clf()

plt.figure(figsize=(12,7))
plt.errorbar(ls_list, hyp_dict['4k'], std_dict['4k'], marker='o', label='sinusoid (4k)', color='blue', capsize=4)
plt.hlines(ref_dict['4k'][0], ls_list[0], ls_list[-1], label='reference', color='grey', linestyle='dashed')
plt.errorbar(ls_list, hyp_dict['pos_learned_4k'], std_dict['pos_learned_4k'], marker='o', label='learned', color='orange', capsize=4)
plt.errorbar(ls_list, hyp_dict['no_pos_4k'], std_dict['no_pos_4k'], marker='o', label='no_pos_4k', color='green', capsize=4)
plt.errorbar(ls_list, hyp_dict['rel_pos_4k'], std_dict['rel_pos_4k'], marker='o', label='rel_pos_4k', color='red', capsize=4)
plt.hlines(training_avg, ls_list[0], ls_list[-1], label='training set', color='grey', linestyle='dotted')

plt.grid()
plt.title('Sampling average sentence length for different positional embeddings (with SD)')
plt.ylabel('length')
plt.xlabel('label smoothing alpha')
plt.legend()
plt.savefig('img/pos_sampling_4k_std.png', dpi=400)
plt.clf()

