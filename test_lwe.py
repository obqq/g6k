#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import sys
import copy
from time import time

from math import log, ceil, floor, exp


from functools import wraps

from g6k.utils.ntru_hybrid_estimation import plain_hybrid_complexity, ntru_plain_hybrid_basis
from g6k.utils.ntru_gsa import find_beta

from g6k.utils.cli import parse_args, run_all, pop_prefixed_params

from ntru_hybrid import read_lwe_from_file, reduction, bdd_query_plain_hybrid, bdd_query_mitm


TESTS_BASEDIR = 'lwe_tests'


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print(f.__name__, te - ts)
        print('func: took: %2.4f sec' % (te - ts))
        return result
    return wrap


def write_to_csv(data, filename):
    file_path = os.path.join(TESTS_BASEDIR, f'{filename}.csv')

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(sorted(data, key=lambda x: len(x.keys()))[-1])
        for d in data:
            w.writerow(d.values())


def ntru_kernel(n, params):
    # params for underlying BKZ
    extra_dim4free = params.get("bkz/extra_dim4free")
    jump = params.get("bkz/jump")
    dim4free_fun = params.get("bkz/dim4free_fun")
    pump_params = pop_prefixed_params("pump", params)
    fpylll_crossover = params.get("bkz/fpylll_crossover")
    blocksizes = params.get("bkz/blocksizes")
    tours = params.get("bkz/tours")

    print("params:", extra_dim4free, jump, dim4free_fun, pump_params,
        fpylll_crossover, blocksizes, tours)

    # flow of the lwe solver
    svp_bkz_time_factor = params.get("lwe/svp_bkz_time_factor")
    goal_margin = params.get("lwe/goal_margin")

    # generation of lwe instance and Kannan's embedding

    # misc
    dont_trace = params.get("dummy_tracer")
    verbose = params.get("verbose")

    #
    # Loading pregenerated LWE instance
    #

    A, b, s, e, q = read_lwe_from_file(n)

    print("Hybrid attack on LWE n=%d" % n)

    # compute the attack parameters
    w = 2*(n/3.)
    paramset_NTRU1 = {'n': n, 'q': q, 'w': w}
    print('paramset_NTRU', paramset_NTRU1)

    #
    # Preprocessing
    #

    beta, g, rt, nsamples, GSA = plain_hybrid_complexity(paramset_NTRU1, verbose = True)
    print('beta, g, rt, nsamples predicted:', beta, g, rt, nsamples)

    # if g is too small to help, recompute BKZ params
    """
    if g <= 4:
        g = 0
        beta, nsamples,rt, GSA = find_beta(n, q, n)
    """
    #force g = 6 for testing the hybrid

    print(type(g))

    if n < 50 or g == 0:
        g = 6

    g = 6

    beta, nsamples,rt, GSA = find_beta(n, q, n)
    print('beta, g, rt, nsamples:', beta, g, rt, nsamples)
    # print('GSA predicted:')
    # print([exp(GSA[i]) for i in range(len(GSA))])

    return A, b, s, q, g, beta, nsamples



def test_babai(B, Ag, b, g, n, q):
    ell = n - g

    b = copy.copy(b)
    b.transpose()

    V1 = []

    M = GSO.Mat(B)
    M.update_gso()

    k = 0
    for sg in gen_secret(g):
        if k % 10000 == 0:
            print(k)
        k += 1

        sg = IntegerMatrix.from_matrix([sg])

        sA = sg * Ag
        target = [0]*(ell+n)
        for i in range(n):
            target[i + ell] = (sA[0][i] - b[i][0]) % q

        v = M.babai(target)
        v = CVP.closest_vector(B, target)

        error = [(target[i] - v[i]) for i in range(n + ell)]




def test_beta(A, b, s, q, g, n, beta, mitm=True):
    g = 6

    B, Al, Ag = ntru_plain_hybrid_basis(A, g, q, nsamples)
    # B = ntru_basis(A, g, q, nsamples, b)
    # blocksizes = list(range(10, 50)) + [beta-20, beta-17] + list(range(beta - 14, beta + 25, 2))
    # print("blocksizes:", blocksizes)


    test_data = []
    for beta in range(0, 100, 10):
        B = reduction(B, beta, params)


        print(g6k.M.get_r(0, 0))


        iter_data = {'beta': beta, }
        test_data.append(iter_data)
    write_to_csv(test_data, 'test_beta')

    # if g == 0:
    #     if(g6k.M.get_r(0, 0) <= target_norm):
    #         return g6k.M.B[0]
    #     else:
    #         raise ValueError("No solution found.")


    # n = A.ncols
    # ell = n - g
    #
    # d = 100 # simhash distance
    #
    #
    # if mitm:
    #     s_ = bdd_query_mitm(B, Ag, b, g, n, q, d, beta, params, list(s[0]))
    # else:
    #     s_ = bdd_query_plain_hybrid(B, Ag, b, g, n, q, beta, params)
    #
    # if s_ is None:
    #     raise ValueError("No solution found.")



def test_plain(arg0, params=None, seed=None):
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    test_data = []

    for n in [16, 20, 24, 32, 48, 64, 100]: #  128, 256
        t1 = time()
        A, b, s, q, g, beta, nsamples = ntru_kernel(n, params)
        B, Al, Ag = ntru_plain_hybrid_basis(A, g, q, nsamples)
        B = reduction(B, beta, params)

        # bdd_query_mitm(B, Ag, b, g, n, q, beta, list(s[0]))
        bdd_query_plain_hybrid(B, Ag, b, g, n, q)

        t2 = time()
        tt = "%.2f" % (t2 - t1)
        print(tt)
        iter_data = {'n': n, 'beta': beta, 'g': g, 'time': tt}
        test_data.append(iter_data)
    write_to_csv(test_data, f'test_plain_{q}')


def test_base(arg0, params=None, seed=None):
    # A, b, s, q, g, beta = ntru_kernel(n, params, seed)
    # test_beta(A, b, s, q, g, n, beta, mitm=True)
    test_plain(arg0, params, seed)



def ntru():
    """
    Attempt to solve an lwe challenge.

    """
    description = ntru.__doc__


    args, all_params = parse_args(description,
                                  ntru__m=None,
                                  lwe__goal_margin=1.5,
                                  lwe__svp_bkz_time_factor=1,
                                  bkz__blocksizes=None,
                                  bkz__tours=1,
                                  bkz__jump=1,
                                  bkz__extra_dim4free=12,
                                  bkz__fpylll_crossover=51,
                                  bkz__dim4free_fun="default_dim4free_fun",
                                  pump__down_sieve=True,
                                  dummy_tracer=True,  # set to control memory
                                  verbose=True
                                  )

    stats = run_all(test_base, list(all_params.values()), # noqa
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


def main():
    ntru()


if __name__ == '__main__':
    main()
