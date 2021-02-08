import sys
import numpy as np
import pickle

from scipy.stats import entropy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix, save_npz, load_npz

np.random.seed(0)

def list_minimum_group(li, is_sensitive):
    return np.bincount(is_sensitive[li], minlength=2).min()

def list_entropy(li, is_sensitive):
    a = np.bincount(is_sensitive[li], minlength=2)
    return entropy(a / a.sum(), base=2)

def select_list(score, is_sensitive, used, K, b):
    assert(b * 2 <= K)
    score = score.copy()
    score[used] -= score.max() + 1
    li = []
    cnt = [0, 0]
    for x in score.argsort()[::-1]:
        cur_sensitive = int(is_sensitive[x])
        if cnt[1 - cur_sensitive] + K - len(li) <= b:
            continue
        cnt[cur_sensitive] += 1
        li.append(x)
        if len(li) == K:
            break
    return np.array(li)

X = np.load('adult_X.npy')
y = np.load('adult_y.npy')
is_sensitive = np.load('adult_a.npy')

m, d = X.shape
K = 10

weight = 1 / np.log2(np.arange(K)[::-1] + 2)
weight /= weight.sum()

R = np.load('adult_R.npy')
At = load_npz('adult_At.npz')
rank = np.load('adult_rank.npy')

cs = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
bs = [0, 1, 2, 3, 4, 5]

platform_accuracy = 0
oracle_accuracy = [0 for j in bs]
ours_accuracy = [[0 for j in bs] for i in cs]
rw_accuracy = [0 for j in bs]
random_accuracy = [0 for j in bs]

platform_minimum = 0
platform_entropy = 0
oracle_minimum = [0 for j in bs]
oracle_entropy = [0 for j in bs]
ours_minimum = [[0 for j in bs] for i in cs]
ours_entropy = [[0 for j in bs] for i in cs]
rw_minimum = [0 for j in bs]
rw_entropy = [0 for j in bs]
random_minimum = [0 for j in bs]
random_entropy = [0 for j in bs]

for i in range(m):
    source = i
    used = [source]
    
    list_platform = rank[source]

    platform_accuracy += (y[list_platform] == y[source]).sum()
    platform_minimum += list_minimum_group(list_platform, is_sensitive)
    platform_entropy += list_entropy(list_platform, is_sensitive)


    for k, b in enumerate(bs):
        oracle_list = []
        cnt = [0, 0]
        for j in np.argsort(R[source])[::-1]:
            if j not in used + oracle_list and cnt[1 - int(is_sensitive[j])] + K - len(oracle_list) > b:
                oracle_list.append(j)
                cnt[int(is_sensitive[j])] += 1
            if len(oracle_list) == K:
                break
        oracle_list = np.array(oracle_list)
        oracle_accuracy[k] += (y[oracle_list] == y[source]).sum()
        oracle_minimum[k] += list_minimum_group(oracle_list, is_sensitive)
        oracle_entropy[k] += list_entropy(oracle_list, is_sensitive)


    for j, c in enumerate(cs):
        psc = np.zeros(m)
        cur = np.zeros(m)
        cur[source] = 1
        for _ in range(11):
            psc += (1 - c) * cur
            cur = c * At @ cur
        for k, b in enumerate(bs):
            ours_list = select_list(psc, is_sensitive, used, K, b)
            ours_accuracy[j][k] += (y[ours_list] == y[source]).sum()
            ours_minimum[j][k] += list_minimum_group(ours_list, is_sensitive)
            ours_entropy[j][k] += list_entropy(ours_list, is_sensitive)
    
    for k, b in enumerate(bs):
        rw_list = []
        cnt = [0, 0]
        for l in range(K):
            cur = source
            max_length = 100
            for _ in range(max_length):
                cur = np.random.choice(rank[cur], p=weight)
                cur_sensitive = int(is_sensitive[cur])
                if cur not in used + rw_list and cnt[1 - cur_sensitive] + K - l > b:
                    break
            while cur in used + rw_list or cnt[1 - cur_sensitive] + K - l <= b:
                cur = np.random.randint(m)
                cur_sensitive = int(is_sensitive[cur])
            rw_list.append(cur)
            cnt[cur_sensitive] += 1
        rw_list = np.array(rw_list)
        rw_accuracy[k] += (y[rw_list] == y[source]).sum()
        rw_minimum[k] += list_minimum_group(rw_list, is_sensitive)
        rw_entropy[k] += list_entropy(rw_list, is_sensitive)


    random_score = np.random.rand(m)
    for k, b in enumerate(bs):
        random_list = select_list(random_score, is_sensitive, used, K, b)
        random_accuracy[k] += (y[random_list] == y[source]).sum()
        random_minimum[k] += list_minimum_group(random_list, is_sensitive)
        random_entropy[k] += list_entropy(random_list, is_sensitive)


    print(i)
    print('plat accuracy:', platform_accuracy)
    print('orcl accuracy:', oracle_accuracy)
    print('ours accuracy:', ours_accuracy)
    print('rndw accuracy:', rw_accuracy)
    print('rand accuracy:', random_accuracy)

    print('plat minimum:', platform_minimum)
    print('orcl minimum:', oracle_minimum)
    print('ours minimum:', ours_minimum)
    print('rndw minimum:', rw_minimum)
    print('rand minimum:', random_minimum)

    print('plat entropy:', platform_entropy)
    print('orcl entropy:', oracle_entropy)
    print('ours entropy:', ours_entropy)
    print('rndw entropy:', rw_entropy)
    print('rand entropy:', random_entropy)
