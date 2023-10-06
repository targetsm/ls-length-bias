import sys
import matplotlib.pyplot as plt
import math


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
    return total_len, count, total_len/count, math.sqrt(variance/count)/math.sqrt(count)


ls_list = ['0', '0.001', '0.005', '0.01', '0.05', '0.1']

#reference length
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
    _, _, mean, se = stats(f'samples/8-3-1000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/8-100-1000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)

fig, axs = plt.subplots(1,3)
fig.set_size_inches(12, 7)
fig.suptitle('Differing number of reference bpe tokens per sample')
axs[0].errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='3 tokens')
axs[0].set_ylim(1, 5)
axs[0].set_title('3 tokens')
axs[0].grid()

axs[1].errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='10 tokens')
axs[1].set_ylim(8, 12)
axs[1].set_title('10 tokens (base)')
axs[1].grid()

axs[2].errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='100 tokens')
axs[2].set_title('100 tokens')
axs[2].grid()

fig.supxlabel('label smoothing alpha')
fig.supylabel('Mean sample length')

plt.savefig('ref_len.png', dpi=400)
plt.clf()

#runs
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
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_3gram_v2.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_3gram_v3.txt')
    v3.append(mean)
    v3_se.append(se)

plt.ylim(8, 12)
plt.grid()
plt.title('Mean length of sampled sentences for different runs of the base model')
plt.ylabel('Mean sample length')
plt.xlabel('label smoothing alpha')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='run 1 (base)')
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='run 2')
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='run 3')
plt.legend()
plt.savefig('runs.png', dpi=400) 
plt.clf()

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
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_4gram.txt')
    n4gram.append(mean)
    n4gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_5gram.txt')
    n5gram.append(mean)
    n5gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_8gram.txt')
    n8gram.append(mean)
    n8gram_se.append(se)
    _, _, mean, se = stats(f'samples/8-10-1000_{i}_10gram.txt')
    n10gram.append(mean)
    n10gram_se.append(se)

plt.ylim(8, 12)
plt.grid()
plt.title('Mean length of sampled sentences for different n')
plt.ylabel('Mean sample length')
plt.xlabel('label smoothing alpha')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='3gram (base)')
plt.errorbar(ls_list, n4gram, n4gram_se, capsize=4, marker='o', label='4gram')
plt.errorbar(ls_list, n5gram, n5gram_se, capsize=4, marker='o', label='5gram')
plt.errorbar(ls_list, n8gram, n8gram_se, capsize=4, marker='o', label='8gram')
plt.errorbar(ls_list, n10gram, n10gram_se, capsize=4, marker='o', label='10gram')
plt.legend()
plt.savefig('ngram.png', dpi=400)
plt.clf()

# dict size
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
    _, _, mean, se = stats(f'samples/3-10-1000_{i}.txt')
    v2.append(mean)
    v2_se.append(se)
    _, _, mean, se = stats(f'samples/16-10-1000_{i}.txt')
    v3.append(mean)
    v3_se.append(se)

plt.ylim(8, 12)
plt.grid()
plt.title('Different dictionary sizes (without <s> and </s>)')
plt.ylabel('Mean sample length')
plt.xlabel('label smoothing alpha')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='8 (base)')
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label='3')
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label='16')
plt.legend()
plt.savefig('dict.png', dpi=400)

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
plt.title('Different number of reference set samples')
plt.ylabel('Mean sample length')
plt.xlabel('label smoothing alpha')
plt.errorbar(ls_list, orig, orig_se, capsize=4, marker='o', label='1000 (base)')
plt.errorbar(ls_list, v2, v2_se, capsize=4, marker='o', label="10'000")
plt.errorbar(ls_list, v3, v3_se, capsize=4, marker='o', label="100'000")
plt.legend()
plt.savefig('num_samples.png', dpi=400)

plt.clf()
