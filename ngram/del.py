from typing import Tuple, List, Dict
import os
import numpy as np
from bisect import bisect
import sys
import argparse
import pickle
import torch
from tqdm import tqdm
import pdb
import glob

class TrieNode(object):
    def __init__(self, word: int, max_idx: int):
        # index of the node word
        self.word = word
        self.children = {}
        # 1+the maximum possible word index
        self.max_idx = max_idx
        # Is it the last character of the word.`
        self.is_leaf = False
        # How many times this word appeared in the addition process
        self.counter = 1
        # possible next words
        self.outcomes = None
        # normalized next word log-distribution
        self.log_distribution = None

    def normalize_leaves(self, smooth_alpha=-np.inf):
        self.outcomes = list(self.children.keys())
        counts = np.array([(lambda x: self.children[i].counter if i in self.outcomes else 0)(i) for i in range(self.max_idx)])
        unnorm_log_distribution = np.log(counts)# + smooth_alpha) 
        #unnorm_log_distribution[unnorm_log_distribution == -np.inf] = np.log(smooth_alpha)
        self.log_distribution = lognormalize(unnorm_log_distribution)

def lognormalize(x: np.array) -> np.array:
    a = np.logaddexp.reduce(x)
    return (x - a)


def add(root, ngram: List[str], token_to_idx_map: Dict[str, int]) -> None:
    """
    Adding an ngram in the trie structure
    """

    node = root
    for i, word in enumerate(ngram):
        word_id = token_to_idx_map.get(word, 3)
        # Search for the character in the children of the present `node`
        if word_id in node.children:
            # And point the node to the child that contains this char
            node = node.children[word_id]
        # We did not find it so add a new chlid
        else:
            new_node = TrieNode(word_id, len(token_to_idx_map))
            node.children[word_id] = new_node
            # And then point node to the new child
            node = new_node

    # Everything finished. Mark it as the end of a word.
    node.counter += 1
    node.is_leaf = True


def find_prefix(root: TrieNode, prefix: List[str], token_to_id_map: Dict[str, int]) -> Tuple[List[int], np.array]:
    """
    Check and return 
      1. If the prefix exsists in any of the words we added so far
      2. If yes, then the possible next words and their log-distribution
    """
    node = root
    for word in prefix:
        try:
            node = node.children[token_to_id_map[word]]
        except KeyError:
            return None, ([np.log(0)]*root.max_idx)
            #return None, ([np.log(1/root.max_idx)] * root.max_idx)
    
    return node.outcomes, node.log_distribution


def normalize(root: TrieNode, smooth_alpha=-np.inf):
    for word, child in root.children.items():
        if child.children[next(iter(child.children))].is_leaf:
            child.normalize_leaves(smooth_alpha)
        else:
            normalize(child, smooth_alpha)


def sample(root: TrieNode, orig_prefix: List[str], token_to_id_map: Dict[str, int], eps:float, n:int) -> str:
    id_to_token_map = {v: k for k, v in token_to_id_map.items()}
    sentence = []
    prefix = orig_prefix
    cur_word = prefix[-1]
    while cur_word != "</s>":
        outcomes, log_probs = find_prefix(root, prefix, token_to_id_map)
        log_probs_smoothed = np.log((1-eps) * np.exp(log_probs) + eps * (1/root.max_idx))
        sampled_idx = log_multinomial_sample(log_probs_smoothed)
        cur_word = id_to_token_map[sampled_idx]
        sentence.append(cur_word)
        prefix = prefix[1:] + [cur_word]
    return sentence


def log_multinomial_sample(x: np.array) -> int:
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    returns the (index of) the sampled word
    """ 
    renormalized = lognormalize(x)
    return np.random.choice(a=list(range(len(x))), p=np.exp(renormalized))
    
    #print('c', c)
    #key = np.log(np.random.uniform())+c[-1]
    #return bisect(c, key)


def get_ngrams(sentence: str, n: int=5) -> List[List[str]]:
    words = sentence.split()
    words = ["<s>"]*(n-1) + words + ["</s>"]
    return [words[i:i+n] for i in range(len(words)-n+1)] 


def create_model(sentences: List[str], n: int, dictionary: List[str], token_to_idx_map: Dict[str, int], smooth_alpha=-np.inf, eps=0) -> TrieNode:
    words = set()

    for l in dictionary:
        words.add(token_to_idx_map[l.split()[0]])
    root = TrieNode(len(dictionary), len(dictionary))
    i=0
    for s in sentences:
        ngrams = get_ngrams(s.strip(), n)
        i+=1
        for ngram in ngrams:
            if all(token_to_idx_map.get(w, 3) in words for w in ngram):
                add(root, ngram, token_to_idx_map)

    normalize(root, smooth_alpha)
    return root


def parse_fairseq_vocabulary(file_path: str) -> List[str]:
    with open(file_path) as f:
        return [l.strip().split(" ")[0] for l in f.readlines()]


def parse_dataset(file_path: str, line_i: int=0) -> List[str]:
    """
    Parses a file having a dataset sample per line
    """
    with open(file_path) as f:
        return [l.strip() for l in f.readlines()[line_i:]]


def get_vocabulary_idx_maps(vocab: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab_idx = {word: i for i, word in enumerate(vocab)}

    return vocab_idx, {v: k for k, v in vocab_idx.items()}


def next_word_probabilities(root: TrieNode, context_ngram: List[str], token_to_idx_map: Dict[str, int]) -> np.array:
    outcomes, next_word_logs = find_prefix(root, context_ngram, token_to_idx_map)

    if outcomes is None or next_word_logs is None:
        return np.array([-np.inf for x in range(len(token_to_idx_map))])

    return next_word_logs


def get_sample_prob_matrix(root: TrieNode, sentence: str, token_to_idx_map: Dict[str, int]) -> torch.FloatTensor:
    ngrams = get_ngrams(sentence)
    probs = [next_word_probabilities(root, ngram[:-1], token_to_idx_map) for ngram in ngrams]

    return torch.tensor(np.stack(probs)) 


def add_special_tokens_to_vocabulary(vocab: List[str]) -> List[str]:
    return ["<s>", "<pad>", "</s>", "<unk>"] + vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=['generate', 'sample'])
    parser.add_argument('-n', '--ngrams', type=int, default=3, help="Number of ngrams")
    parser.add_argument('--data_path', type=str, help="Path to file containing processed data. One sample per line")
    parser.add_argument('--dict_path', type=str, help="Path to dict.txt from fairseq-preprocess")
    parser.add_argument('--output_path', type=str, help="Where to store the sampled sentences")
    parser.add_argument('--model_path', type=str, help="Path to new/existing ngram")
    parser.add_argument('--smooth_alpha', type=float, default=-1.0, help="Degree of smoothing, -1 will be converted to -np.inf for a sparse model")
    parser.add_argument('--ls_eps', type=float, default=0.0, help="Factor of label smoothing")

    args = parser.parse_args()
    sentences = parse_dataset(args.data_path)    
    dictionary = parse_fairseq_vocabulary(args.dict_path)

    n = args.ngrams
    smooth_alpha = args.smooth_alpha
    if args.smooth_alpha == -1.0:
        smooth_alpha = -np.inf
    print(f"SMOOTHNESS: {smooth_alpha}")
    
    eps = args.ls_eps
    print(f"LS_EPS: {eps}")

    dictionary = add_special_tokens_to_vocabulary(dictionary)
    token_to_idx, _ = get_vocabulary_idx_maps(dictionary)
      
    if args.task == 'generate':
        root = create_model(sentences, n, dictionary, token_to_idx_map=token_to_idx, smooth_alpha=smooth_alpha)
        print("CREATED MODEL")
        pickle.dump(root, open(args.model_path, "wb"))
        pickle.dump(token_to_idx, open(args.model_path + '.vocab', 'wb'))
    elif args.task == 'sample':
        root = pickle.load(open(args.model_path, "rb"))
        with open(args.output_path, 'w') as f:
            for i in range(100000):
                sampled = sample(root, ['<s>']*(n-1), token_to_idx, eps, n)
                f.write(' '.join(sampled[:-1])+'\n')

