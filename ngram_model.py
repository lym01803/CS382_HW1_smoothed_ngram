import math
import numpy as np
from tqdm import tqdm 

class NGramModel:
    def __init__(self, n):
        self.n = n

    def build(self, **kwargs):
        raise NotImplementedError
    
    def get_log_P(self, tokens):
        raise NotImplementedError

class MacKayNGramModel(NGramModel):
    def __init__(self, n=2):
        assert (n > 1)
        super().__init__(n)
        self._F = dict()
        self._Fj = dict()
        self._G = dict()
        self._H = dict()
        self._V = dict()
        self._vocab = dict()
        self._u = dict()
        self.alpha = 2.0
        self._lambda_j = dict()

    def build(self, **kwargs):
        text = kwargs['text']
        n = self.n
        for i in tqdm(range(n - 1, len(text))):
            self._add(text[i - n + 1: i + 1])
        for i in tqdm(self._vocab):
            self._calc_i(i)
        self._calc_alpha_m()
        for j in tqdm(self._Fj):
            self._lambda_j[j] = self.alpha / (self.alpha + self._Fj[j])
        
    def _calc_i(self, i):
        G_i = 0.0
        H_i = 0.0
        js = self._vocab[i]
        js.sort()
        counts = []
        count = 0
        for idx in range(len(js)):
            if idx and js[idx] != js[idx - 1]:
                counts.append(count)
                count = 0
            count += 1
        counts.append(count)
        counts.sort(key=lambda x: -x)
        F_max = counts[0]
        V_i = len(counts)
        for f in range(2, F_max + 1):
            while counts[-1] < f:
                counts.pop()
            G_i += len(counts) / (f - 1.0)
            H_i += len(counts) / (f - 1.0) / (f - 1.0)
        self._G[i] = G_i
        self._H[i] = H_i 
        self._V[i] = V_i
    
    def _K(self, alpha):
        summ = 0.0
        for j in tqdm(self._Fj):
            F_j = self._Fj[j]
            summ += math.log((F_j + alpha) / alpha)
            summ += 0.5 * F_j / (alpha * (F_j + alpha))
        return summ

    def _calc_alpha_m(self, eps=1e-3):
        self.alpha = 2.0
        iter = 0
        while True:
            print(f'iter: {iter}')
            new_alpha = 0.0
            Ka = self._K(self.alpha)
            for i in tqdm(self._vocab):
                Ka_Gi = Ka - self._G[i]
                new_ui = 2.0 * self._V[i] / (
                    Ka_Gi + math.sqrt(Ka_Gi ** 2 + 4.0 * self._H[i] * self._V[i])
                )
                new_alpha += new_ui
                self._u[i] = new_ui 
            if math.fabs(self.alpha - new_alpha) < eps:
                break 
            self.alpha = new_alpha
            iter += 1
    
    def _add(self, tokens):
        tokens = tuple(tokens)
        if tokens in self._F:
            self._F[tokens] += 1
        else:
            self._F[tokens] = 1
        i = tokens[-1]
        j = tokens[:-1]
        if j in self._Fj:
            self._Fj[j] += 1
        else:
            self._Fj[j] = 0
        if i in self._vocab:
            self._vocab[i].append(j)
        else:
            self._vocab[i] = [j]
    
    def get_log_P(self, tokens):
        i = tokens[-1]
        j = tokens[:-1]
        lambda_j = self._lambda_j[j]
        f_i_j = self._F.get(tokens, 0)
        if f_i_j:
            f_i_j /= self._Fj[j]
        return math.log(
            lambda_j * self._u.get(i, 0) / self.alpha + (1 - lambda_j) * f_i_j
        )

def count_ngrams(fd, n):
    corpus = fd.read().strip().split()
    _set = set()
    for i in tqdm(range(n - 1, len(corpus))):
        _set.add(tuple(corpus[i - n + 1: i + 1]))
    return len(_set)

if __name__ == '__main__':
    paths = ['./hw1_dataset/train_set.txt', './hw1_dataset/dev_set.txt']
    corpus = []
    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            corpus += f.read().strip().split()
    model = MacKayNGramModel()
    model.build(text=corpus)
    test_path = './hw1_dataset/test_set.txt'
    with open(test_path, 'r', encoding='utf8') as f:
        test_text = f.read().strip.split()
    
