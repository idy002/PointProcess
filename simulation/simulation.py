import math
from random import random
from math import pow, e, log

U = [0.001, 0, 0.1, 0.005, 0.007, 0.0025, 0.003, 0.0069, 0.0081, 0.0043]
A = [[0.100, 0.072, 0.0044, 0.000, 0.0023, 0.000, 0.09, 0.000, 0.07, 0.025],
     [0.000, 0.050, 0.068, 0.000, 0.027, 0.065, 0.000, 0.000, 0.097, 0.000],
     [0.093, 0.000, 0.0062, 0.045, 0.000, 0.000, 0.053, 0.0095, 0.000, 0.083],
     [0.019, 0.0033, 0.000, 0.073, 0.058, 0.000, 0.056, 0.000, 0.000, 0.000],
     [0.045, 0.091, 0.000, 0.000, 0.066, 0.000, 0.000, 0.033, 0.0058, 0.000],
     [0.067, 0.000, 0.000, 0.000, 0.000, 0.055, 0.063, 0.078, 0.085, 0.0095],
     [0.000, 0.022, 0.0013, 0.000, 0.057, 0.091, 0.0088, 0.065, 0.000, 0.073],
     [0.000, 0.090, 0.000, 0.088, 0.000, 0.078, 0.000, 0.09, 0.068, 0.000],
     [0.000, 0.000, 0.093, 0.000, 0.033, 0.000, 0.069, 0.000, 0.082, 0.033],
     [0.001, 0.000, 0.089, 0.000, 0.008, 0.000, 0.0069, 0.000, 0.000, 0.072]]
w = 0.6


def simulation(U, A, w, T):
    """
    simulate a M-dimensional hawkes process

    :param U: M dimensional vector, background intensity vector
    :param A: MxM dimensional matrix, the infectivity matrix
    :param w: scalar, decay ratio
    :param T: simulate in time range [0, T]
    :return: a list of time sequence, the length of list if M
    """
    M = len(U)
    sequences = [[] for _ in range(M)]

    def calc_lambda(s, m):
        """
        calculate the m-dimension intensity on time s
        :param s: time
        :param m: dimension
        :return: the intensity
        """
        return U[m] + sum([A[n][m] * pow(e, w * (s - hs)) for n in range(M) for hs in sequences[n]])

    s = 0.0
    while s <= T:
        lambda_max = sum([calc_lambda(s, m) for m in range(M)])
        u = 1.0 - random()  # (0, 1]
        s += -log(u) / lambda_max
        if s > T: break
        print(s)
        u = 1.0 - random()
        lambda_list = [calc_lambda(s, m) for m in range(M)]
        if u * lambda_max <= sum(lambda_list):
            k = 0
            while k < M:
                if u * lambda_max <= sum(lambda_list[:k + 1]):
                    break
                k += 1
            sequences[k].append(s)
    return sequences


seqs = simulation(U, A, w, 1000.0)
for seq in seqs:
    print(seq)
