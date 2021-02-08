import sys
import numpy as np
import pickle

from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import entropy
from implicit.nearest_neighbours import CosineRecommender

np.random.seed(0)

def recall(li, gt):
    if gt in li:
        return 1
    return 0

def nDCG(li, gt):
    if gt in li:
        return 1 / np.log2(li.tolist().index(gt) + 2)
    return 0

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

assert(len(sys.argv) == 4)

data = sys.argv[1]
if data == '100k':
    n = 943
    m = 1682
    filename = 'ml-100k/u.data'
    delimiter = '\t'
elif data == '1m':
    n = 6040
    m = 3952
    filename = 'ml-1m/ratings.dat'
    delimiter = '::'

K = 10

if data == '100k' or data == '1m':
    raw_R = np.zeros((n, m))
    history = [[] for i in range(n)]
    with open(filename) as f:
        for r in f:
            user, movie, r, t = map(int, r.split(delimiter))
            user -= 1
            movie -= 1
            raw_R[user, movie] = r
            history[user].append((t, movie))
elif data == 'hetrec':
    raw_R = np.log2(np.load('hetrec.npy') + 1)
    n, m = raw_R.shape
    history = [[] for i in range(n)]
    for i in range(n):
        for j in np.nonzero(raw_R[i] > 0)[0]:
            history[i].append((np.random.rand(), j))
elif data == 'home':
    raw_R = np.load('Home_and_Kitchen.npy')
    n, m = raw_R.shape
    with open('Home_and_Kitchen_history.pickle', 'br') as f:
        history = pickle.load(f)

platform_method = sys.argv[2]
sensitive_attribute = sys.argv[3]
if sensitive_attribute == 'popularity':
    mask = raw_R > 0
    if data == '100k':
        is_sensitive = mask.sum(0) < 50
    elif data == '1m':
        is_sensitive = mask.sum(0) < 300
    elif data == 'hetrec':
        is_sensitive = mask.sum(0) < 30
    elif data == 'home':
        is_sensitive = mask.sum(0) < 30
elif sensitive_attribute == 'old':
    is_sensitive = np.zeros(m, dtype='bool')
    if data == '100k':
        filename = 'ml-100k/u.item'
        delimiter = '|'
    elif data == '1m':
        filename = 'ml-1m/movies.dat'
        delimiter = '::'
    with open(filename) as f:
        for r in f:
            l = r.strip().split(delimiter)
            if '(19' in l[1]:
                year = 1900 + int(l[1].split('(19')[1].split(')')[0])
            elif '(20' in l[1]:
                year = 2000 + int(l[1].split('(20')[1].split(')')[0])
            is_sensitive[int(l[0])-1] = year < 1990

cs = [0.01]
bs = [0, 1, 2, 3, 4, 5]

platform_recall = 0
platform_nDCG = 0
oracle_recall = [0 for j in bs]
oracle_nDCG = [0 for j in bs]
ours_recall = [[0 for j in bs] for i in cs]
ours_nDCG = [[0 for j in bs] for i in cs]
rw_recall = [0 for j in bs]
rw_nDCG = [0 for j in bs]
random_recall = [0 for j in bs]
random_nDCG = [0 for j in bs]


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

for i in range(n):
    gt = sorted(history[i])[-1][1]
    source = sorted(history[i])[-2][1]
    used = [y for x, y in history[i] if y != gt]

    R = raw_R.copy()
    R[i, gt] = 0

    mask = R > 0

    if platform_method == 'common':
        score = mask.astype('float64').T @ mask
    else:
        if platform_method == 'bpr':
            model = BayesianPersonalizedRanking(num_threads=1, random_state=0)
        elif platform_method == 'als':
            model = AlternatingLeastSquares(num_threads=1)
        elif platform_method == 'cosine':
            model = CosineRecommender()

        sR = csr_matrix(mask.T)
        model.fit(sR)
        if platform_method == 'bpr':
            score = model.item_factors @ model.item_factors.T
        else:
            score = np.zeros((m, m))
            for item in range(m):
                for j, v in model.similar_items(item, m):
                    score[item, j] = v

    score_remove = score.copy()
    score_remove[:, used] -= 1e9
    score_remove -= np.eye(m) * 1e9

    list_platform = np.argsort(score_remove[source])[::-1][:K]

    platform_recall += recall(list_platform, gt)
    platform_nDCG += nDCG(list_platform, gt)
    platform_minimum += list_minimum_group(list_platform, is_sensitive)
    platform_entropy += list_entropy(list_platform, is_sensitive)

    for k, b in enumerate(bs):
        oracle_list = []
        cnt = [0, 0]
        for j in np.argsort(score_remove[source])[::-1]:
            if j not in used + oracle_list and cnt[1 - int(is_sensitive[j])] + K - len(oracle_list) > b:
                oracle_list.append(j)
                cnt[int(is_sensitive[j])] += 1
            if len(oracle_list) == K:
                break
        oracle_list = np.array(oracle_list)
        oracle_recall[k] += recall(oracle_list, gt)
        oracle_nDCG[k] += nDCG(oracle_list, gt)
        oracle_minimum[k] += list_minimum_group(oracle_list, is_sensitive)
        oracle_entropy[k] += list_entropy(oracle_list, is_sensitive)

    A = np.zeros((m, m))
    rank = np.argsort(score_remove, 1)[:, -K:]
    weight = 1 / np.log2(np.arange(K)[::-1] + 2)
    weight /= weight.sum()
    A[np.arange(m).repeat(K), rank.reshape(-1)] += weight.repeat(m).reshape(K, m).T.reshape(-1)

    for j, c in enumerate(cs):
        psc = np.zeros(m)
        cur = np.zeros(m)
        cur[source] = 1
        for _ in range(11):
            psc += (1 - c) * cur
            cur = c * A.T @ cur
        for k, b in enumerate(bs):
            ours_list = select_list(psc, is_sensitive, used, K, b)
            ours_recall[j][k] += recall(ours_list, gt)
            ours_nDCG[j][k] += nDCG(ours_list, gt)
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
        rw_recall[k] += recall(rw_list, gt)
        rw_nDCG[k] += nDCG(rw_list, gt)
        rw_minimum[k] += list_minimum_group(rw_list, is_sensitive)
        rw_entropy[k] += list_entropy(rw_list, is_sensitive)


    random_score = np.random.rand(m)
    for k, b in enumerate(bs):
        random_list = select_list(random_score, is_sensitive, used, K, b)
        random_recall[k] += recall(random_list, gt)
        random_nDCG[k] += nDCG(random_list, gt)
        random_minimum[k] += list_minimum_group(random_list, is_sensitive)
        random_entropy[k] += list_entropy(random_list, is_sensitive)


    print(i)
    print('plat recall:', platform_recall)
    print('orcl recall:', oracle_recall)
    print('ours recall:', ours_recall)
    print('rndw recall:', rw_recall)
    print('rand recall:', random_recall)
    print('plat nDCG:', platform_nDCG)
    print('orcl nDCG:', oracle_nDCG)
    print('ours nDCG:', ours_nDCG)
    print('rndw nDCG:', rw_nDCG)
    print('rand nDCG:', random_nDCG)

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

