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
        text = kwargs['text'] + ['<unk>'] * self.n
        n = self.n
        for i in tqdm(range(n - 1, len(text))):
            self._add(text[i - n + 1 : i + 1])
        for i in tqdm(self._vocab):
            self._calc_i(i)
        self._calc_alpha_m()
        for j in tqdm(self._Fj):
            self._lambda_j[j] = self.alpha / (self.alpha + self._Fj[j])
            # print(j, self._lambda_j)
        
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
        for j in self._Fj:
            F_j = self._Fj[j]
            summ += math.log((F_j + alpha) / alpha)
            summ += 0.5 * F_j / (alpha * (F_j + alpha))
        return summ

    def _calc_alpha_m(self, eps=1e-6):
        self.alpha = 2.0
        iter = 0
        while True:
            new_alpha = 0.0
            Ka = self._K(self.alpha)
            for i in self._vocab:
                Ka_Gi = Ka - self._G[i]
                new_ui = 2.0 * self._V[i] / (
                    Ka_Gi + math.sqrt(Ka_Gi ** 2 + 4.0 * self._H[i] * self._V[i])
                )
                new_alpha += new_ui
                self._u[i] = new_ui 
            if math.fabs(self.alpha - new_alpha) < eps:
                break 
            print(f'iter: {iter}, delta: {new_alpha - self.alpha}')
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
            self._Fj[j] = 1
        if i in self._vocab:
            self._vocab[i].append(j)
        else:
            self._vocab[i] = [j]
    
    def get_log_P(self, tokens, eps=math.exp(-99)):
        tokens = tuple(token if token in self._vocab else '<unk>' for token in tokens)
        assert(len(tokens) == self.n)
        i = tokens[-1]
        j = tokens[:-1]
        lambda_j = self._lambda_j.get(j, 1.0)
        f_i_j = self._F.get(tokens, 0)
        if f_i_j:
            f_i_j /= self._Fj[j]
        # lambda_j = 0.0
        p = lambda_j * self._u.get(i, 0) / self.alpha + (1 - lambda_j) * f_i_j
        return math.log(max(p, eps))

class InterpolationNGramModel(NGramModel):
    def __init__(self, n):
        assert (n > 1)
        super().__init__(n)
        self._F = dict()
        self._P = dict()
        self._W = dict()

    def build(self, **kwargs):
        train = kwargs['train'] + ['<unk>'] * self.n
        dev = kwargs['dev']
        for n in range(1, self.n + 1):
            for i in tqdm(range(n - 1, len(train))):
                self._add(train[i - n + 1 : i + 1])
        count = len(train)
        for i in tqdm(range(len(train))):
            if not i in self._P:
                self._P[i] = self._F.get(i, 0) / count
        self.smoothing(dev)
        
    def _add(self, tokens):
        tokens = tuple(tokens)
        if tokens in self._F:
            self._F[tokens] += 1
        else:
            self._F[tokens] = 1

    def smoothing(self, dev, eps=1e-3):
        n = self.n
        for n in range(2, self.n + 1):
            self._smoothing_n(n, dev, eps)
    
    def get_P(self, tokens):
        tokens = tuple(token if token in self._F else '<unk>' for token in tokens)
        p = self._P.get(tokens, -1)
        if p < 0:
            if len(tokens) > 1:
                p = self._W.get(tokens[:-1], 1.0) * self.get_P(tokens[1:])
            else:
                p = 1e-30
        return p

    def get_log_P(self, tokens):
        return math.log(self.get_P(tokens))

    def _smoothing_n(self, n, dev, eps=1e-3):
        counts = []
        for i in range(n - 1, len(dev)):
            counts.append(self._F.get(tuple(dev[i - n + 1 : i]), 0))
        counts = list(set(counts))
        counts.sort()
        count2idx = dict()
        for idx, count in enumerate(counts):
            count2idx[count] = idx
        L = [0.0 for count in counts]
        R = [1.0 for count in counts]
        res = [0.0 for count in counts]
        while True:
            for i in tqdm(range(n - 1, len(dev))):
                tokens = tuple(dev[i - n + 1 : i + 1])
                idx = count2idx[self._F.get(tokens[:-1], 0)]
                m_i = self.get_P(tokens[1:])
                lambda_idx = (L[idx] + R[idx]) * 0.5
                f_i_j = self._F.get(tokens, 0)
                if f_i_j:
                    f_i_j /= self._F.get(tokens[:-1])
                res[idx] += (m_i - f_i_j) / (lambda_idx * m_i + (1.0 - lambda_idx) * f_i_j)
            max_interval = 0.0
            for idx in range(len(counts)):
                if res[idx] > 0:
                    L[idx] = (L[idx] + R[idx]) * 0.5
                else:
                    R[idx] = (L[idx] + R[idx]) * 0.5
                max_interval = max(max_interval, R[idx] - L[idx])
            print(max_interval)
            if max_interval < eps:
                break
        for tokens in self._F:
            if len(tokens) == n - 1:
                j = tokens
                idx = count2idx.get(j, 0)
                self._W[j] = L[idx] # log(lambda_idx * m_i + 0) = log(lambda_idx) + log(m_i); log back-off weight
            if len(tokens) == n:
                j, i = tokens[:-1], tokens[-1]
                idx = count2idx.get(j, 0)
                lambda_idx = L[idx]
                f_i_j = self._F.get(tokens) / self._F.get(j)
                self._P[tokens] = lambda_idx * self.get_P(tokens[1:]) + (1.0 - lambda_idx) * f_i_j

def count_ngrams(fd, n):
    corpus = fd.read().strip().split()
    _set = set()
    for i in tqdm(range(n - 1, len(corpus))):
        _set.add(tuple(corpus[i - n + 1: i + 1]))
    return len(_set)

def calc_ppl(text, model:NGramModel):
    n = model.n
    token_num = 0
    log_p_sum = 0.0
    for i in range(n - 1, len(text)):
        log_p_sum += model.get_log_P(text[i - n + 1 : i + 1])
        token_num += 1
    avg_log_p = log_p_sum / token_num
    return math.exp(-avg_log_p)

if __name__ == '__main__':
    '''
    paths = ['./hw1_dataset/train_set.txt', './hw1_dataset/dev_set.txt']
    corpus = []
    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            corpus += f.read().strip().split()
    model = MacKayNGramModel(2)
    model = MacKayNGramModel(2)
    model.build(text=corpus)
    '''
    paths = ['./hw1_dataset/train_set.txt', './hw1_dataset/dev_set.txt']
    with open(paths[0], 'r', encoding='utf8') as f:
        train = ['<s>'] + f.read().strip().split() + ['<s/>']
    with open(paths[1], 'r', encoding='utf8') as f:
        dev = ['<s>'] + f.read().strip().split() + ['<s/>']
    model = InterpolationNGramModel(2)
    model.build(train=train, dev=dev)
    test_path = './hw1_dataset/test_set.txt'
    with open(test_path, 'r', encoding='utf8') as f:
        test_text = ['<s>'] + f.read().strip().split() + ['<s/>']
    print(calc_ppl(test_text, model))
    
