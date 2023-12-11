import sys
import matplotlib.pyplot as plt
import math
import numpy as np
from sacrebleu.metrics import BLEU#, CHRF, TER

ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
hyp_dict = {'base':[], '4k':[], '64k':[]}
bleu_dict = {'base':[], '4k':[], '64k':[]}
ref_dict = {'base':[], '4k':[], '64k':[]}
ref_corpus = []
hyp_corpus = []

for d in hyp_dict.keys():
    for l in ls_list:
        f = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/{d}/evaluation/ls_{l}/generate-test.txt','r').readlines()
        hyp_len = 0
        ref_len = 0
        total_count = 0
        ref_corpus = []
        hyp_corpus = []
        bleu = BLEU()
        for line in f:
            if line[0] == 'H':
                hyp_len += len(line.split()[2:])
                hyp_corpus.append(''.join(line.split()[2:]).replace('▁',' '))
            elif line[0] == 'T':
                ref_corpus.append(''.join(line.split()[1:]).replace('▁',' '))
                ref_len += len(line.split()[1:])
                total_count += 1
        bleu_dict[d].append(bleu.corpus_score(hyp_corpus, [ref_corpus]).score)
        hyp_dict[d].append(hyp_len / total_count)
        #hyp_dict[d].append(hyp_len / ref_len)
        print(ref_len, total_count)
        ref_len = ref_len / total_count
        ref_dict[d].append(ref_len)
        print(hyp_dict)
        print(bleu_dict)
        print(ref_len)

plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(3.9*1.2,1.7*1.2))
#plt.ylim(0.5, 1.2)

plt.plot(ls_list, hyp_dict['4k'], '-o', label='4k', color='orange')
plt.hlines(ref_dict['4k'][0], ls_list[0], ls_list[-1], color='orange', label='4k ref', linestyle='dashed')
plt.plot(ls_list, hyp_dict['base'], '-o', label='16k', color='blue')
plt.hlines(ref_dict['base'][0], ls_list[0], ls_list[-1], color='blue', label='16k ref', linestyle='dashed')
plt.plot(ls_list, hyp_dict['64k'], '-o', label='64k', color='green')
plt.hlines(ref_dict['64k'][0], ls_list[0], ls_list[-1], color='green', label='64k ref', linestyle='dashed')
plt.grid()
#plt.title('Beam search mean sentence length for different bpe dictionary sizes')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/length.pdf', dpi=400)
plt.clf()

plt.figure(figsize=(3.9*1.2,1.7*1.2))
plt.plot(ls_list, bleu_dict['4k'], '-o', label='4k', color='orange')
plt.plot(ls_list, bleu_dict['base'], '-o', label='16k', color='blue')
plt.plot(ls_list, bleu_dict['64k'], '-o', label='64k', color='green')
plt.grid()
#plt.title('Beam search BLEU for different bpe dictionary sizes')
plt.ylabel('BLEU')
plt.xlabel('label smoothing $\\lambda$')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('img/bleu.pdf', dpi=400)

