#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NTRU
"""

from __future__ import absolute_import
from __future__ import print_function
import re
import sys
import time
import copy

import os.path

from collections import OrderedDict # noqa
from math import log, ceil, floor, exp

import numpy as np

import numpy as np

from fpylll import GSO, IntegerMatrix, BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from six.moves import range

from g6k.utils.ntru_hybrid_estimation import plain_hybrid_complexity, ntru_plain_hybrid_basis
from g6k.utils.ntru_gsa import find_beta

from simhash import SimHashes, search, XPC_WORD_LEN


NTRU_BASEDIR = 'ntru_challenge'
LWE_BASEDIR = 'lwe_challenge'


def read_ntru_from_file(n):
    file_path = os.path.join(NTRU_BASEDIR, f'ntru_n_{n}.txt')

    if not os.path.isfile(file_path):
        raise ValueError (f'File {file_path} not found!')

    data = open(file_path, "r").readlines()
    q = int(data[0])
    H = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[1 :]]))
    H = IntegerMatrix.from_matrix(H)
    return H, q


def read_lwe_from_file(n):
    file_path = os.path.join(LWE_BASEDIR, f'lwe_n_{n}.txt')

    if not os.path.isfile(file_path):
        raise ValueError (f'File {file_path} not found!')

    data = open(file_path, "r").readlines()
    q = int(data[0])
    A = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[1:-3]]))
    A = IntegerMatrix.from_matrix(A)
    b = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[-3:-2]]))
    b = IntegerMatrix.from_matrix([b])
    s = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[-2:-1]]))
    s = IntegerMatrix.from_matrix([s])
    e = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[-1:]]))
    e = IntegerMatrix.from_matrix([e])
    return A, b, s, e, q


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

#
# TODO: generalise to different number of 1s and -1s
#
def kbits(n, k):
    """
    create all combinations of length k of indicies {0,...n-1}
    these are all possible positions of "1"s

    k position of "1"s
    k position of "-1"s
    """
    assert 2*k <= n
    full_range = range(n)

    # todo: add for all possible number of -1, 0, 1,
    for pos1 in combinations(full_range, k):
        # remove pos1 from the range of pos2
        new_range = []
        start_pos = 0
        for puncture in pos1:
            new_range += tuple(range(start_pos, puncture))
            start_pos = puncture+1
        new_range += tuple(range(start_pos, n))
        # all possible positions of "-1"s
        for pos2 in combinations(new_range, k):
            yield pos1, pos2


def bdd_query(B, Ag, b, g, n, q, d=1000):
    '''
    Batch-CVP (BDD) with preprocessing, or (batch-CVPP).
    Using Babai's Nearest Plane
    '''
    SH = SimHashes(n, seed=1337)

    ell = n - g

    V1 = []
    V2 = []

    b = copy.copy(b)
    b.transpose()

    M = GSO.Mat(B) # B
    M = GSO.Mat(B)
    M.update_gso()

    # len = 1681680
    k = 0
    poss = kbits(g, ceil(g / 3))
    all_pos = len(list(poss))
    for pos1, pos2 in kbits(g, ceil(g / 3)):
        if k % 10000 == 0:
            print(k, all_pos - k)
        k += 1

        s = [0] * g
        for i in pos1:
            s[i] = 1
        for i in pos2:
            s[i] = -1

        s1 = IntegerMatrix.from_matrix([s[:g // 2] + [0] * (g // 2)])

        target = s1 * Ag
        target.mod(q)
        target = [0] * ell + list(target[0])

        v1 = M.babai(target)
        # print(target, v1)

        V1.append((list(v1), s[:g // 2]))
        V2.append(s[g // 2:])


    # Closest Pairs

    # V1 = np.array([np.array(SH.compress(v) + v) for v in V1])
    V1 = np.array([np.array(SH.compress(v) + v + s) for v, s in V1])

    # sorting by multiple columns tests:
    # https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    V1 = V1[np.lexsort([V1[:,i] for i in range(XPC_WORD_LEN, -1, -1)])]

    print(V1[0])

    for k, s2 in enumerate(V2):
        if k % 1000 == 0:
            print(k, all_pos - k)

        # BDD:
        # s2_ = IntegerMatrix.from_iterable(g, 1, s[g // 2:] + [0] * (g // 2))
        s2_ = IntegerMatrix.from_matrix([s2 + [0] * (g // 2)])

        target =  s2_ * Ag
        b_ = copy.copy(b)
        b_[0] -= target[0]
        target = IntegerMatrix.from_matrix(b_)
        target.transpose()

        target.mod(q)
        target = [0] * ell + list(target[0])

        v2 = M.babai(target)

        # Search:

        v2_hash = SH.compress(v2)
        close_vec = search(V1, v2_hash, d=10000)
        if close_vec is not None: # todo
            print(i)
            v1, v1_hash, s1 = close_vec[XPC_WORD_LEN:-(g // 2)], close_vec[:XPC_WORD_LEN], close_vec[-(g // 2):]
            print(v1, v1_hash, s1)
            print(v2, v2_hash, s2)
            # print((v2, v2_hash), (close_vec[XPC_WORD_LEN:], close_vec[:XPC_WORD_LEN]))
            sg = list(s1) + list(s2)
            return sg


def ntru_kernel(arg0, params=None, seed=None):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/goal_margin: accept anything that is
          goal_margin * estimate(length of embedded vector)
          as an lwe solution

        - lwe/svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the lwe challenge attempt

    """

    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    # params for underlying BKZ
    extra_dim4free = params.pop("bkz/extra_dim4free")
    jump = params.pop("bkz/jump")
    dim4free_fun = params.pop("bkz/dim4free_fun")
    pump_params = pop_prefixed_params("pump", params)
    fpylll_crossover = params.pop("bkz/fpylll_crossover")
    blocksizes = params.pop("bkz/blocksizes")
    tours = params.pop("bkz/tours")

    print("params:", extra_dim4free, jump, dim4free_fun, pump_params,
        fpylll_crossover, blocksizes, tours)

    # flow of the lwe solver
    svp_bkz_time_factor = params.pop("lwe/svp_bkz_time_factor")
    goal_margin = params.pop("lwe/goal_margin")

    # generation of lwe instance and Kannan's embedding

    # misc
    dont_trace = params.pop("dummy_tracer")
    verbose = params.pop("verbose")

    # A, q = read_ntru_from_file(n)
    A, b, s, e, q = read_lwe_from_file(n)


    print("Hybrid attack on NTRU n=%d" % n)


    # compute the attack parameters
    w = 2*(n/3.)
    paramset_NTRU1 = {'n': n, 'q': q, 'w': w}
    print(paramset_NTRU1)
    beta, g, rt, nsamples, GSA = plain_hybrid_complexity(paramset_NTRU1, verbose = True)
    print('beta, g, rt, nsamples:', beta, g, rt, nsamples)

    # if g is too small to help, recompute BKZ params
    if g <= 4:
        g = 0
        beta, nsamples,rt, GSA = find_beta(n, q, n)

    g = 12

    print('beta, g, rt, nsamples:', beta, g, rt, nsamples)
    print('GSA predicted:')
    print([exp(GSA[i]) for i in range(len(GSA))])

    # fails for g = 0 (n = 32, 64)
    B, Al, Ag = ntru_plain_hybrid_basis(A, g, q, nsamples)

    # blocksizes = list(range(10, 50)) + [beta-20, beta-17] + list(range(beta - 14, beta + 25, 2))
    # print("blocksizes:", blocksizes)

    g6k = Siever(B, params)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("ntru"), start_clocks=True)

    d = g6k.full_n
    g6k.lll(0, g6k.full_n)
    print(g6k.MatGSO)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    print('GSA input:')
    print([g6k.M.get_r(i, i) for i in range(d)])

    print('d:', d)
    target_norm = ceil( (2./3)*d + 1) + 1
    print("target_norm:", target_norm)

    # beta = 50
    #
    #   Preprocessing
    #
    if beta < fpylll_crossover:
        print("Starting a fpylll BKZ-%d tour. " % (beta), end=' ')
        sys.stdout.flush()
        bkz = BKZReduction(g6k.M)
        par = fplll_bkz.Param(beta,
                              strategies=fplll_bkz.DEFAULT_STRATEGY,
                              max_loops=1)
        bkz(par)

    else:
        print("Starting a pnjBKZ-%d tour. " % (beta))
        pump_n_jump_bkz_tour(g6k, tracer, beta, jump=jump,
                                 verbose=verbose,
                                 extra_dim4free=extra_dim4free,
                                 dim4free_fun=dim4free_fun,
                                 goal_r0=target_norm,
                                 pump_params=pump_params)

            #T_BKZ = time.time() - T0_BKZ

    print('GSA output:')
    print([g6k.M.get_r(i, i) for i in range(d)])
    #print(g6k.M.get_r(0, 0))
    print(g6k.M.B[0], g6k.M.get_r(0,0))

    if g == 0:
        if(g6k.M.get_r(0, 0) <= target_norm):
            return g6k.M.B[0]
        else:
            raise ValueError("No solution found.")

    print(g6k.M.B, B)


    n = A.ncols
    ell = n - g
    print(n, g, ell)
    # todo: coeffs like in ntru_plain_hybrid_basis

    print(A)
    print()
    print(Al)
    print()
    print(Ag)

    # BDD Queries

    sg = bdd_query(B, Ag, b, g, n, q)

    # check = b - e - sg * Ag

    print(s)
    print(sg)

    if sg is None:
        raise ValueError("No solution found.")


def ntru(n):
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

    stats = run_all(ntru_kernel, list(all_params.values()), # noqa
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


def main():
    # read_lwe_from_file(64)
    _, n = sys.argv
    ntru(int(n))

if __name__ == '__main__':
    main()
