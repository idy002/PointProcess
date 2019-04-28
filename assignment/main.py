import os
import json
import argparse
import numpy as np
from simulation import simulation, draw_qq_plot, draw_line
from fit import fit


def evaluation(real_parameters, fitted_parameters):
    """
    return the mean relative error
    """
    U1, U2 = real_parameters['U'], fitted_parameters['U']
    A1, A2 = real_parameters['A'], fitted_parameters['A']
    w1, w2 = real_parameters['w'], fitted_parameters['w']
    X1, X2 = [], []
    X1.extend(U1)
    X2.extend(U2)
    for a1, a2 in zip(A1, A2):
        X1.extend(a1)
        X2.extend(a2)
    if w1 != w2:
        X1.append(w1)
        X2.append(w2)
    X1, X2 = np.array(X1), np.array(X2)
    return np.sum(np.where(X1 == 0, np.abs(X1 - X2), np.abs(X1 - X2) / X1))


def main():
    """
    simulate a multi-dimensional hawkes point process, the parameters are hard encoded
    """
    # parse argument
    parser = argparse.ArgumentParser("Multi-dimensional hawkes point process simulator")
    parser.add_argument('params', type=str, nargs='?', default='parameters.json', help="the filepath of json parameter file")
    args = parser.parse_args()

    # load parameters
    parameters = json.load(open(args.params, "rt"))

    # simulate
    U, A, w, T = parameters['U'], parameters['A'], parameters['w'], parameters['T']
    seqs = simulation(U, A, w, T)

    # save result
    dirname = "result"
    os.makedirs(dirname, exist_ok=True)
    json.dump(parameters, open(os.path.join(dirname, "parameters.json"), "wt"), indent=2)
    json.dump(seqs, open(os.path.join(dirname, "sequences.json"), "wt"), indent=2)

    # draw figures
    draw_line(seqs, os.path.join(dirname, "line.svg"))
    draw_qq_plot(parameters, seqs, os.path.join(dirname, "qq_plot.svg"))

    # fit
    fitted_parameters = fit(seqs, T, w)
    print("Mean relative error: {:.5f}".format(evaluation(parameters, fitted_parameters)))


main()

