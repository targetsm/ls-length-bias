import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


for s in  ['_norep']:
    paths = ['sample_5gram_4k'+s, 'sample_3gram_4k'+s, 'sample_3gram_16k'+s]
    ref = "iwslt17.de-en.bpe4k/train.bpe.de-en.en"
    gen_list = ['ls_0.txt', 'ls_0.001.txt', 'ls_0.005.txt', 'ls_0.01.txt', 'ls_0.05.txt',
            'ls_0.1.txt']#,  'ls_0.25.txt', 'ls_0.5.txt', 'ls_1.txt']
    plot_labels = ['0', '0.001', '0.005', '0.01', '0.05',
            '0.1']#, '0.25', '0.5', '1']
    lr_fig, lr_ax = plt.subplots()
    mean_fig, mean_ax = plt.subplots()
    
    for path in paths:
        if path == 'sample_3gram_16k'+s:
            ref = "iwslt17.de-en.bpe16k/train.bpe.de-en.en"
        else:
            ref = "iwslt17.de-en.bpe4k/train.bpe.de-en.en"
        lr_list = []
        mean_list = []
        for gen in gen_list:
        
            len_ref = 0
            total_len_ref = 0
            total_len_gen = 0
            with open(ref) as r:
                for line in r:
                    len_ref += len(line.split())
                    total_len_ref += 1
            len_gen = 0
            with open(path + '/' + gen) as g:
                for line in g:
                    len_gen += len(line.split())
                    total_len_gen += 1
            print(ref, 'vs.', path + '/' + gen)
            print('len_ref:', len_ref)
            print('len_gen:', len_gen)
            print('len_ref/total_len_ref:', len_ref/total_len_ref)
            if path == 'sample_3gram_16k'+s:
                ref_mean_16k = len_ref/total_len_ref
            else:
                ref_mean_4k = len_ref/total_len_ref
            #ref = "iwslt17.de-en.bpe4k/train.bpe.de-en.en"
            print('len_gen/total_len_gen:', len_gen/total_len_gen)
            print('(len_gen/total_len_gen) / (len_ref/total_len_ref):', (len_gen/total_len_gen) / (len_ref/total_len_ref))
            print('(len_gen/total_len_gen) - (len_ref/total_len_ref):', (len_gen/total_len_gen) - (len_ref/total_len_ref))
            lr_list.append((len_gen/total_len_gen) / (len_ref/total_len_ref))
            mean_list.append(len_gen/total_len_gen)
        lr_ax.plot(plot_labels, lr_list, label=path[:-6])
        mean_ax.plot(plot_labels, mean_list, label=path[:-6])
    #lr_ax.set_yticks(np.arange(0, 1.3, 0.1))
    lr_ax.legend()
    lr_ax.set_xlabel('label smoothing alpha')
    lr_ax.set_ylabel('length ratio')
    lr_ax.set_yscale('log')
    lr_ax.set_yticks([1, 1.2,1.5, 2, 10, 100])
    lr_ax.yaxis.set_ticklabels([1,1.2, 1.5, 2, 10, 100])
    lr_ax.yaxis.set_major_formatter(ScalarFormatter())    
    lr_ax.grid()
    lr_ax.set_title('ratio of mean sample length to mean reference length')
    
    mean_ax.set_xlabel('label smoothing alpha')
    mean_ax.set_ylabel('mean length')
    mean_ax.set_yscale('log')
    mean_ax.set_yticks([20, 30, 100, 1000])
    mean_ax.yaxis.set_major_formatter(ScalarFormatter())
    mean_ax.grid()
    mean_ax.hlines(ref_mean_16k, 0, len(plot_labels)-1, label='ref_16k', linestyles='dashed', colors='green')
    mean_ax.hlines(ref_mean_4k, 0, len(plot_labels)-1, label='ref_4k', linestyles='dashed')
    mean_ax.legend()

    mean_ax.set_title('Mean sample and reference lengths')
    

    lr_fig.savefig(f'lr_plot{s}.png')
    mean_fig.savefig(f'mean_plot{s}.png')
