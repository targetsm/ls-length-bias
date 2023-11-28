import sys
import matplotlib.pyplot as plt
import math

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'base':[], 'pos_learned':[], 'no_pos':[], 'rel_pos':[]}
bleu_dict = {'base':[], 'pos_learned':[], 'no_pos':[], 'rel_pos':[]}
ref_dict = {'base':[], 'pos_learned':[], 'no_pos':[], 'rel_pos':[]}
se_dict = {'base':[], 'pos_learned':[], 'no_pos':[], 'rel_pos':[]}
std_dict = {'base':[], 'pos_learned':[], 'no_pos':[], 'rel_pos':[]}

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
            hyp_se += math.sqrt(sum([(x - hyp_sum/length)**2 for x in ls[2:]])/length)/math.sqrt(length) * 2.576
            hyp_std += math.sqrt(sum([(x - hyp_sum/length)**2 for x in ls[2:]])/length)
            total_count += 1
        hyp_dict[d].append(hyp_len/total_count)
        se_dict[d].append(hyp_se / total_count)
        std_dict[d].append(hyp_std / total_count)
        ref_dict[d].append(ref_len / total_count)
        print(hyp_dict[d], se_dict[d], ref_dict[d])
plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(3.9*1.2,1.7*1.2))
plt.errorbar(ls_list, hyp_dict['base'], se_dict['base'], marker='o', label='sinusoid', color='blue', capsize=4)
plt.errorbar(ls_list, hyp_dict['pos_learned'], se_dict['pos_learned'], marker='o', label='learned', color='orange', capsize=4)
plt.errorbar(ls_list, hyp_dict['no_pos'], se_dict['no_pos'], marker='o', label='none', color='green', capsize=4)
plt.errorbar(ls_list, hyp_dict['rel_pos'], se_dict['rel_pos'], marker='o', label='relative', color='red', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], label='test', color='grey', linestyle='dashed')


training_avg = 0
training_file = open('../data/iwslt17.de-en.bpe16k/train.bpe.de-en.en').readlines()
for line in training_file:
    training_avg += len(line.split())
training_avg /= len(training_file)
plt.hlines(training_avg, ls_list[0], ls_list[-1], label='train', color='grey', linestyle='dotted')

plt.grid()
#plt.title('Sampling mean sentence length for different positional embeddings (with SE)')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/pos_sampling.pdf', dpi=400)
plt.clf()

plt.figure(figsize=(3.9*1.2,1.7*1.2))
plt.errorbar(ls_list, hyp_dict['base'], std_dict['base'], marker='o', label='sinusoid', color='blue', capsize=4)
plt.errorbar(ls_list, hyp_dict['pos_learned'], std_dict['pos_learned'], marker='o', label='learned', color='orange', capsize=4)
plt.errorbar(ls_list, hyp_dict['no_pos'], std_dict['no_pos'], marker='o', label='none', color='green', capsize=4)
plt.errorbar(ls_list, hyp_dict['rel_pos'], std_dict['rel_pos'], marker='o', label='relative', color='red', capsize=4)
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], label='test', color='grey', linestyle='dashed')
plt.hlines(training_avg, ls_list[0], ls_list[-1], label='train', color='grey', linestyle='dotted')

plt.grid()
#plt.title('Sampling mean sentence length for different positional embeddings (with SD)')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/pos_sampling_std.pdf', dpi=400)
plt.clf()

