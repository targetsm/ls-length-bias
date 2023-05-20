import sys
import matplotlib.pyplot as plt
import numpy as np

for s in  ['', '_norep']:
    paths = ['sample_5gram_4k'+s, 'sample_3gram_4k'+s, 'sample_3gram_16k'+s]
    ref = "iwslt17.de-en.bpe4k/train.bpe.de-en.en"
    gen_list = ['no_ls.txt', 'ls_0.001.txt', 'ls_0.01.txt',
            'ls_0.1.txt', 'ls_0.25.txt', 'ls_0.5.txt', 'ls_1.txt']
    plot_labels = ['0', '0.001', '0.01',
            '0.1', '0.25', '0.5', '1']
    lr_fig, lr_ax = plt.subplots()
    bias_fig, bias_ax = plt.subplots()
    
    for path in paths:
        if path == 'sample_3gram_16k'+s:
            ref = "iwslt17.de-en.bpe16k/train.bpe.de-en.en"
        else:
            ref = "iwslt17.de-en.bpe4k/train.bpe.de-en.en"
        lr_list = []
        bias_list = []
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
            print('len_gen/total_len_gen:', len_gen/total_len_gen)
            print('(len_gen/total_len_gen) / (len_ref/total_len_ref):', (len_gen/total_len_gen) / (len_ref/total_len_ref))
            print('(len_gen/total_len_gen) - (len_ref/total_len_ref):', (len_gen/total_len_gen) - (len_ref/total_len_ref))
            lr_list.append((len_gen/total_len_gen) / (len_ref/total_len_ref))
            bias_list.append((len_gen/total_len_gen) - (len_ref/total_len_ref))
        lr_ax.plot(plot_labels, lr_list, label=path)
        bias_ax.plot(plot_labels, bias_list, label=path)
    #lr_ax.set_yticks(np.arange(0, 1.3, 0.1))
    lr_ax.legend()
    lr_ax.set_xlabel('label smoothing alpha')
    lr_ax.set_ylabel('length ratio (train set)')
    lr_ax.grid()
    lr_ax.hlines(1, 0, len(plot_labels)-1, colors='black')
    lr_ax.set_title('length ratio')
    
    #bias_ax.set_yticks(np.arange(-20, 5, 1))
    bias_ax.legend()
    bias_ax.set_xlabel('label smoothing alpha')
    bias_ax.set_ylabel('bias (train set)')
    bias_ax.grid()
    bias_ax.hlines(0, 0, len(plot_labels)-1, colors='black')
    bias_ax.set_title('bias')
    
    lr_fig.savefig(f'lr_plot{s}.png')
    bias_fig.savefig(f'bias_plot{s}.png')
