from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import sys

src = open(sys.argv[1], 'r').readlines()
trg = open(sys.argv[2], 'r').readlines()


new_src = []
new_trg = []
for i in range(len(src)):
    src_tok = word_tokenize(src[i])
    trg_tok = word_tokenize(trg[i])
    overlap = 0
    for word in src_tok:
        if word in trg_tok:
            overlap += 1
    if overlap < len(src_tok)/2:
        new_src.append(src[i])
        new_trg.append(trg[i])
    #else:
        #print(src[i], trg[i])

print(f'Removed {len(src) - len(new_src)} partial copies from src')
print(f'Removed {len(trg) - len(new_trg)} partial copies from trg')


with open(sys.argv[1], 'w') as out_src:
    out_src.writelines(new_src)
with open(sys.argv[2], 'w') as out_trg:
    out_trg.writelines(new_trg)

