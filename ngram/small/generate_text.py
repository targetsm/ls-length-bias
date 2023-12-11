import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dict_size', type=int, default=8)
parser.add_argument('--sent_len', type=int, default=10)
parser.add_argument('--corpus_size', type=int, default=10000)

args = parser.parse_args()

#simple_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
simple_dict = [str(x) for x in range(0,args.dict_size)]

with open(f'data/{args.dict_size}-{args.sent_len}-{args.corpus_size}.txt', 'w') as f:
    for i in range(args.corpus_size):
        sentence = []
        for j in range(args.sent_len):
            sentence.append(np.random.choice(simple_dict))
        f.write(' '.join(sentence) + '\n')

with open(f'data/{args.dict_size}-{args.sent_len}-{args.corpus_size}.dict', 'w') as f:
    for i in simple_dict:
        f.write(i + ' 0\n')
