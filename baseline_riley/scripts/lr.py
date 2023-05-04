import sys

hyp = open(sys.argv[1]).readlines()
ref = open(sys.argv[2]).readlines()

hyp_len = 0
for sent in hyp:
    hyp_len += len(' '.join(sent.split()))
    #print(len(sent))
ref_len = 0
for sent in ref:
    ref_len += len(' '.join(sent.split()))
    #print(len(sent))
print('paper_lr:', hyp_len/ref_len)

ratio_sum = 0
for i in range(len(hyp)):
    ratio_sum += len(' '.join(hyp[i].split()))/len(' '.join(ref[i].split()))

print('normal_lr:', ratio_sum/len(hyp))

