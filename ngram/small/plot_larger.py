import sys
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter

font = {'family' : 'sans serif'}
#        'size'   : 22}

matplotlib.rc('font', **font)

def stats(f):
    f = open(f).readlines()


    total_len = 0
    count = 0
    for line in f:
        total_len += len(line.split())
        count += 1
    mean = total_len/count

    variance=0
    for line in f:
        variance += (len(line.split())-mean)**2
    return total_len, count, total_len/count, math.sqrt(variance/count)/math.sqrt(count)*2.576


ls_list = ['0', '0.001', '0.005', '0.01', '0.05', '0.1']

#reference length
orig = []
orig_se = []
v2 = []
v2_se = []
v3 = []
v3_se = []
for i in ls_list:
    _, _, mean, se = stats(f'samples/8-10-10000_{i}.txt')
    orig.append(mean)
    orig_se.append(se)
    _, _, mean, se = stats(f'samples/8-3-10000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/8-100-10000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)
plt.rcParams.update({'figure.autolayout': True})
fig, axs = plt.subplots(3,1, layout='constrained', sharex=True)
fig.set_size_inches(4, 5)
#fig.suptitle('Mean sampling length for different bpe vocabulary sizes')
axs[0].errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='3')
axs[0].set_ylim(1, 5)
#axs[0].set_title('3 tokens')
axs[0].legend()
axs[0].grid()

axs[1].errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='10')
axs[1].set_ylim(8, 12)
#axs[1].set_title('10 tokens')
axs[1].legend()
axs[1].grid()

axs[2].errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='100')
#axs[2].set_title('100 tokens')
axs[2].legend()
axs[2].grid()

fig.supxlabel('label smoothing $\\lambda$')
fig.supylabel('length')

plt.savefig('ref_len.pdf', dpi=400)
fig.set_size_inches(3.9,1.7)
plt.clf()

plt.rcParams['figure.constrained_layout.use'] = True

#reference length
orig = []
orig_se = []
v2 = []
v2_se = []
v3 = []
v3_se = []
for i in ls_list:
    _, _, mean, se = stats(f'samples/8-10-10000_{i}.txt')
    orig.append(mean)
    orig_se.append(se)
    _, _, mean, se = stats(f'samples/8-3-10000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/8-100-10000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)
#plt.ylim(8, 12)
plt.grid()
#plt.title('Mean length of sampled sentences for different runs of the base model')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.yscale('log')
ax = plt.gca()
ax.set_yticks([3, 5, 10, 50, 100])
ax.yaxis.set_major_formatter(ScalarFormatter())
#plt.get_yaxis().set_minor_formatter(mticker.ScalarFormatter())
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='3')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='10')
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='100')
#plt.legend()
#plt.legend(bbox_to_anchor=(1.05, 1),
#                         loc='upper left', borderaxespad=0.)
plt.legend(title='$\ell_{\mathrm{toy}}$', bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                      ncols=3, mode="expand", borderaxespad=0.)
plt.savefig('ref_len_small.pdf', dpi=400)
plt.clf()

fig.set_size_inches(3.9,1.7)

# ngrams
n4gram = []
n4gram_se = []
n5gram = []
n5gram_se = []
n8gram = []
n8gram_se = []
n10gram = []
n10gram_se = []
for i in ls_list:
    _, _, mean, se = stats(f'samples/8-10-10000_{i}_4gram.txt')
    n4gram.append(mean)
    n4gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-10000_{i}_5gram.txt')
    n5gram.append(mean)
    n5gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-10000_{i}_8gram.txt')
    n8gram.append(mean)
    n8gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-10000_{i}_10gram.txt')
    n10gram.append(mean)
    n10gram_se.append(se)

plt.ylim(8, 12)
plt.grid()
#plt.title('Mean length of sampled sentences for different n')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='3gram')
plt.errorbar(ls_list, n4gram, n4gram_se, capsize=4, marker='o', label='4gram')
plt.errorbar(ls_list, n5gram, n5gram_se, capsize=4, marker='o', label='5gram')
plt.errorbar(ls_list, n8gram, n8gram_se, capsize=4, marker='o', label='8gram')
plt.errorbar(ls_list, n10gram, n10gram_se, capsize=4, marker='o', label='10gram')
#plt.legend()
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                      ncols=3, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('ngram.pdf', dpi=400)
plt.clf()

# dict size
orig = []
orig_se = []
v2 = []
v2_se = []
v3 = []
v3_se = []
for i in ls_list:
    _, _, mean, se = stats(f'samples/8-10-10000_{i}.txt')
    orig.append(mean)
    orig_se.append(se)
    _, _, mean, se = stats(f'samples/3-10-10000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/16-10-10000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)

plt.ylim(8, 12)
plt.grid()
#plt.title('Different dictionary sizes')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='5')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='10')
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='18')
#plt.legend()
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                      ncols=3, mode="expand", borderaxespad=0.)
plt.legend(title='$|\\overline{\\mathcal{V}}_{\\mathrm{toy}}|$', bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('dict.pdf', dpi=400)

plt.clf()

# number of train set samples
orig = []
orig_se = []
v2 = []
v2_se = []
v3 = []
v3_se = []
for i in ls_list:
    _, _, mean, se = stats(f'samples/8-10-1000_{i}.txt')
    orig.append(mean)
    orig_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-10000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-100000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)

plt.ylim(8, 12)
plt.grid()
#plt.title('Different number of reference set samples')
plt.ylabel('length')
plt.xlabel('label smoothing $\\lambda$')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='1000')
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label="10'000")
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label="100'000")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                      ncols=3, mode="expand", borderaxespad=0.)
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('num_samples.pdf')

plt.clf()
