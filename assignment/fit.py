from math import e, pow
import itertools
import numpy as np
from numpy.linalg import norm


def fit(seqs, T, w=None, max_step=1000):
    """
    inference the multi-hawkes point process parameters
    :param seqs: the list of event sequences, M = len(seqs) is the dimension of the process
    :param T: the data was sampled from time interval [0, T]
    :param w: when w is None, we inference w, otherwise we regard w is known
    :return: parameters, {'U': U, 'A', A, 'w': w}, where U is the background intensity
        and A is the infectivity matrix. A[n][m] is the infectivity of n to m
    """
    events = []
    for index, seq in enumerate(seqs):
        events.extend(zip(seq, itertools.repeat(index)))
    events = sorted(events, key=lambda event: event[0])
    M = len(seqs)
    N = len(events)
    w_known = w is not None
    U = np.random.rand(M)
    A = np.random.rand(M, M)
    if not w_known:
        w = np.random.rand()

    def p(U, A, w, i, j):
        """
        calculate p[i, j] in E-step
        """
        base = U[events[i][1]] + sum([A[events[j][1], events[i][1]] * pow(e, -w * (events[i][0] - events[j][0])) for j in range(i)])
        if i == j:
            return U[events[i][1]] / base
        else:
            return A[events[j][1], events[i][1]] * pow(e, -w * (events[i][0] - events[j][0])) / base

    print("Dimension {}  Events {}".format(M, N))

    step = 0
    while step <= max_step:
        step += 1
        print("Fitting step {} / {}".format(step, max_step))
        old_U = np.copy(U)
        old_A = np.copy(A)
        old_w = np.copy(w)
        pp = np.zeros((N, N), dtype=np.float)
        for i in range(N):
            for j in range(i+1):
                pp[i, j] = p(old_U, old_A, old_w, i, j)
        for d in range(M):
#            U[d] = sum([p(old_U, old_A, old_w, i, i) for i in range(N) if events[i][1] == d]) / T
             U[d] = sum([pp[i, i] for i in range(N) if events[i][1] == d]) / T
        for du in range(M):
            for dv in range(M):
                up, down = 0.0, 0.0
                for i in range(N):
                    if events[i][1] != dv: continue
                    for j in range(i):
                        if events[j][1] != du: continue
                        up += pp[i, j]#p(old_U, old_A, old_w, i, j)
                for j in range(N):
                    if events[j][1] != du: continue
                    down += (1.0 - pow(e, -old_w * (T - events[j][0]))) / old_w
                A[dv, du] = up / down
        if not w_known:
            up, down = 0.0, 0.0
            for i in range(N):
                for j in range(i):
                    pij = pp[i, j]#p(old_U, old_A, old_w, i, j)
                    up += pij
                    down += (events[i][0] - events[j][0]) * pij
            w = up / down
        else:
            w = old_w
        dist = norm(old_U - U) + norm(old_A - A) + norm(old_w - w)
        print("{} {:.5f}".format(step, dist))
        if dist < 1e-5:
            print("Early stop!")
            break
    return {'U': U.tolist(), 'A': A.tolist(), 'w': w}
