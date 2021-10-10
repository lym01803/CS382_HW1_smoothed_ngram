import math
import numpy as np
from tqdm import tqdm 

class NGramModel:
    def __init__(self, n):
        self.n = n
        self._dict = dict()

    def build(self, corpus):
        self._dict.clear()
        n = self.n
        for i in tqdm(range(n - 1, len(corpus))):
            _tuple = tuple(corpus[i - n + 1 : i + 1])
            self._add(_tuple)
    
    def _add(self, _tuple):
        if _tuple in self._dict:
            self._dict[_tuple] += 1
        else:
            self._dict[_tuple] = 1

def count_ngrams(fd, n):
    corpus = fd.read().strip().split()
    _set = set()
    for i in tqdm(range(n - 1, len(corpus))):
        _set.add(tuple(corpus[i - n + 1: i + 1]))
    return len(_set)

if __name__ == '__main__':
    path = './hw1_dataset/train_set.txt'
    fin = open(path, 'r', encoding='utf8')
    print(count_ngrams(fin, 2))
    fin.close()
