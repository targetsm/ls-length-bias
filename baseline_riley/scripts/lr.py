import sys

hyp = open(sys.argv[1]).readlines()
ref = open(sys.argv[2]).readlines()

hyp_len = 0
for sent in hyp:
    hyp_len += len(sent)
    print(len(sent))
ref_len = 0
for sent in ref:
    ref_len += len(sent)
    print(len(sent))
print(hyp_len/ref_len)
