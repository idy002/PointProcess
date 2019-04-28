import math
from math import pow
import itertools
import numpy as np
from numpy.linalg import norm


def fit(seqs, T, w=None, max_step=1000, eps=5e-5):
    """
    inference the multi-hawkes point process parameters
    :param seqs: the list of event sequences, M = len(seqs) is the dimension of the process
    :param T: the data was sampled from time interval [0, T]
    :param w: when w is None, we inference w, otherwise we regard w is known
    :param eps: the epsilon, when the 2-norm of change is less or equal to epsilon, stop iteration
    :return: parameters, {'U': U, 'A', A, 'w': w}, where U is the background intensity
        and A is the infectivity matrix. A[n][m] is the infectivity of n to m
    """
    e = []
    for index, seq in enumerate(seqs):
        e.extend(zip(seq, itertools.repeat(index)))
    e = sorted(e, key=lambda event: event[0])
    M = len(seqs)
    N = len(e)
    w_known = w is not None
    U = np.random.rand(M)
    A = np.random.rand(M, M)
    if not w_known:
        w = np.random.rand()

    print("Dimension {}  Events {}".format(M, N))

    step = 0
    while step <= max_step:
        step += 1
        print("Fitting step {} / {}".format(step, max_step))
        old_U = np.copy(U)
        old_A = np.copy(A)
        old_w = np.copy(w)
        p = np.zeros((N, N), dtype=np.float)
        for i in range(N):
            for j in range(i):
                p[i, j] = A[e[j][1], e[i][1]] * pow(math.e, -w * (e[i][0] - e[j][0]))
            p[i, i] = U[e[i][1]]
            p[i] = p[i] / np.sum(p[i])

        for d in range(M):
             U[d] = sum([p[i, i] for i in range(N) if e[i][1] == d]) / T
        for du in range(M):
            for dv in range(M):
                up, down = 0.0, 0.0
                for i in range(N):
                    if e[i][1] != dv: continue
                    for j in range(i):
                        if e[j][1] != du: continue
                        up += p[i, j]
                for j in range(N):
                    if e[j][1] != du: continue
                    down += (1.0 - pow(math.e, -old_w * (T - e[j][0]))) / old_w
                A[dv, du] = up / down
        if not w_known:
            up, down = 0.0, 0.0
            for i in range(N):
                for j in range(i):
                    pij = p[i, j]
                    up += pij
                    down += (e[i][0] - e[j][0]) * pij
            w = up / down
        else:
            w = old_w
        dist = norm(old_U - U) + norm(old_A - A) + norm(old_w - w)
        print("{} {:.7f}".format(step, dist))
        if dist < eps:
            print("Early stop!")
            break
    return {'U': U.tolist(), 'A': A.tolist(), 'w': w.tolist()}
