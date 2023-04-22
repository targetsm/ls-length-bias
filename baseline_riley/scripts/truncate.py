import sys
import nltk
from nltk.tokenize import WhitespaceTokenizer
     
# Create a reference variable for Class WhitespaceTokenizer
tk = WhitespaceTokenizer()
     
sents = open(sys.argv[1]).readlines()
trunc = float(sys.argv[2])
new_sents = []
for sent in sents:
    text = tk.tokenize(sent)
    new_sents.append(' '.join(text[:round(len(text)*trunc)])+'\n')

with open(sys.argv[1], 'w') as f:
    f.writelines(new_sents)

