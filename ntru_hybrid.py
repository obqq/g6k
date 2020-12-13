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
from itertools import product
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

from fpylll import CVP


NTRU_BASEDIR = 'ntru_challenge'
LWE_BASEDIR = 'lwe_challenge'

def mmodq(v, q):
    n  = len(v)
    vmod = v
    for i in range(n):
        if vmod[i]>(q/2):
            vmod[i] = q - vmod[i]
    return vmod


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


def gen_secret_(n):
    """
    create all combinations of length k of indicies {0,...n-1}
    these are all possible positions of "1"s

    k position of "1"s
    k position of "-1"s
    """
    full_range = range(n)

    # todo: remove duplicates

    for i in range(n): # number of "0"s in this loop
        for pos0 in combinations(full_range, i):

            new_range1 = []
            start_pos = 0
            for puncture in pos0:
                new_range1 += tuple(range(start_pos, puncture))
                start_pos = puncture+1
            new_range1 += tuple(range(start_pos, n))

            for j in range(n - i): # number of "1"s in this loop
                for pos1 in combinations(new_range1, j):

                    # remove pos1 from the range of pos2
                    new_range2 = []
                    start_pos = 0
                    for puncture in pos1:
                        new_range2 += tuple(range(start_pos, puncture))
                        start_pos = puncture+1
                    new_range2 += tuple(range(start_pos, n))
                    # all possible positions of "-1"s
                    for pos2 in combinations(new_range2, n - i - j):

                        yield pos1, pos2


def gen_secret(k):
    values = [-1, 0, 1]
    for s in product(*[values] * k):
        yield list(s)


def test_kbits():
    from pprint import pprint

    g = 3

    secrets = list(gen_secret(g))
    # pprint(positions)

    count = 0
    count_dups = 0
    for i, x in enumerate(secrets):

        for j, y in enumerate(secrets):
            if i != j and x == y:
                count_dups += 1

    print(count_dups, count)
    pprint(sorted(secrets))
    print([0, -1, 0] in secrets)


def check_success(v):
    return all([k in [-1, 0, 1] for k in v])


def reduction(B, beta, params):
    '''
    '''
    extra_dim4free = params.get("bkz/extra_dim4free")
    jump = params.get("bkz/jump")
    dim4free_fun = params.get("bkz/dim4free_fun")
    pump_params = params.get("pump")
    # pump_params = pop_prefixed_params("pump", params)
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

    g6k = Siever(B, params)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("ntru"), start_clocks=True)

    d = g6k.full_n
    g6k.lll(0, g6k.full_n)
    #print(g6k.MatGSO)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    print('GSA input:')
    print([g6k.M.get_r(i, i) for i in range(d)])

    print('d:', d)
    target_norm = ceil( (2./3)*d + 1) + 1
    print("target_norm:", target_norm)

    for tt in range(tours):
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

    print('g6k.M.B')
    print(g6k.M.B)

    #T_BKZ = time.time() - T0_BKZ

    print('GSA output:')
    print([g6k.M.get_r(i, i) for i in range(d)])
    #print(g6k.M.get_r(0, 0))
    print(g6k.M.B[0], g6k.M.get_r(0,0))

    return g6k.M.B


def bdd_query_plain_hybrid(B, Ag, b, g, n, q, beta, params):
    '''
    '''

    SH = SimHashes(n, seed=1337)

    ell = n - g

    b = copy.copy(b)
    b.transpose()

    k = 0
    while beta < 100:
        print(f'beta: {beta}')

        V1 = []

        dim = B.nrows
        target_norm = ceil( (2./3)*dim + 1) + 1

        M = GSO.Mat(B)
        M.update_gso()
        print(B)

        for sg in gen_secret(g):
            if k % 10000 == 0:
                print(k)
            k += 1

            print(sg)

            sg = IntegerMatrix.from_matrix([sg])

            sA = sg * Ag
            target = [0]*(ell+n)
            for i in range(n):
                target[i + ell] = (sA[0][i] - b[i][0]) % q
            # print('target:', target)

            # BABAI MAY NOT BE SUFFICIENT!
            # v = M.babai(target)
            # print('v babai:', v)
            # print(sg, sum([abs(v[i]) for i in range(len(v))]))

            # CVP suffices (but slow)
            v = CVP.closest_vector(B, target)
            # print('v CVP:', v)

            error = [target[i] - v[i] for i in range(n + ell)]
            print('error:', error) # should be +/-1,0
            print(error)
            if check_success(error):
                print('success')
                return sg

        beta += 1
        break

        # makes no difference
        # todo: fix s.t. actually aplies reduction
        B = reduction(B, beta, params)

    raise ValueError("No solution found.")


def bdd_query_mitm(B, Ag, b, g, n, q, d, beta, params):
    '''
    Batch-CVP (BDD) with preprocessing, or (batch-CVPP).
    Using Babai's Nearest Plane
    '''
    ell = n - g
    SH = SimHashes(n + ell, seed=1337)

    V1 = []
    V2 = set()

    b = copy.copy(b)
    b.transpose()

    M = GSO.Mat(B)
    M.update_gso()

    # len = 1681680
    k = 0
    secrets1 = gen_secret(g // 2)
    secrets2 = gen_secret(g // 2)
    # print(all_pos_lst)
    for s1 in secrets1:
        # print(k)
        if k % 1000 == 0:
            print(k)
        k += 1

        s1_ = IntegerMatrix.from_matrix([s1 + [0] * (g // 2)])

        target = s1_ * Ag
        target.mod(q)
        target = [0] * ell + list(target[0])

        # v1 = M.babai(target)
        v1 = CVP.closest_vector(B, target)

        # print(f'v1: {v1}')
        # print(target, v1)

        V1.append((list(v1), s1))


    # Closest Pairs

    V1 = np.array([np.array(SH.compress(v) + v + s) for v, s in V1])

    # sorting by multiple columns tests:
    # https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    V1 = V1[np.lexsort([V1[:,i] for i in range(XPC_WORD_LEN, -1, -1)])]


    for v in V1:
        v1, v1_hash, s1 = v[XPC_WORD_LEN:-(g // 2)], v[:XPC_WORD_LEN], v[-(g // 2):]
        print(v1, v1_hash, s1)

    V1_ = IntegerMatrix.from_matrix([v[:XPC_WORD_LEN].tolist() for v in V1[1:]])
    M_ = GSO.Mat(V1_)
    M_.update_gso()

    k = 0
    for s2 in secrets2:
        # print(k)
        if k % 1000 == 0:
            print(k)
        k += 1

        if s2 == [0, -1, 1]:
            print('found s2')
        else:
            continue

        #
        # BDD:
        #

        s2_ = IntegerMatrix.from_matrix([[0] * (g // 2) + s2])

        sA = s2_ * Ag
        target = [0] * (ell + n)
        for i in range(n):
            target[i + ell] = (sA[0][i] - b[i][0]) % q

        # v2 = M.babai(target)
        v2 = CVP.closest_vector(B, target)

        # Search:

        v2_hash = SH.compress(v2)

        # close_vec = search(V1, v2_hash, d)

        # close_vec_ = CVP.closest_vector(V1_, v2_hash)
        # print(close_vec, close_vec_)

        # close_vec_ = M_.babai(v2_hash)
        # print('babai:', close_vec_)

        for v in V1:
            v1, v1_hash, s1 = v[XPC_WORD_LEN:-(g // 2)], v[:XPC_WORD_LEN], v[-(g // 2):]
            if s1.tolist() == [0, -1, 0]:
                print('found s1')
                break
        close_vec = v

        if close_vec is not None:
            # print(i)
            v1, v1_hash, s1 = close_vec[XPC_WORD_LEN:-(g // 2)], close_vec[:XPC_WORD_LEN], close_vec[-(g // 2):]
            print(close_vec)
            print('v1:', len(v1), v1, v1_hash, s1)
            print('v2:', len(v2), v2, v2_hash, s2)
            # print((v2, v2_hash), (close_vec[XPC_WORD_LEN:], close_vec[:XPC_WORD_LEN]))
            sg = list(s1) + list(s2)
            print(sg)

            v = [0] * (n + ell)
            for i in range(n + ell):
                v[i] = v1[i] + v2[i]

            # error = [target[i] - v[i] for i in range(n + ell)]

            # check = b - e - sg * Ag
            error = [v[i] for i in range(n + ell)]
            print('error:', error) # should be +/-1,0
            if check_success(error):
                print('success')
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
    print(paramset_NTRU1)

    #
    # Preprocessing
    #

    beta, g, rt, nsamples, GSA = plain_hybrid_complexity(paramset_NTRU1, verbose = True)
    print('beta, g, rt, nsamples:', beta, g, rt, nsamples)

    # if g is too small to help, recompute BKZ params
    """
    if g <= 4:
        g = 0
        beta, nsamples,rt, GSA = find_beta(n, q, n)
    """
    #force g = 6 for testing the hybrid
    g = 6
    beta, nsamples,rt, GSA = find_beta(n, q, n)
    print('beta, g, rt, nsamples:', beta, g, rt, nsamples)
    print('GSA predicted:')
    print([exp(GSA[i]) for i in range(len(GSA))])

    B, Al, Ag = ntru_plain_hybrid_basis(A, g, q, nsamples)
    # B = ntru_basis(A, g, q, nsamples, b)
    # blocksizes = list(range(10, 50)) + [beta-20, beta-17] + list(range(beta - 14, beta + 25, 2))
    # print("blocksizes:", blocksizes)

    #
    # First part: Reduction
    #

    # tours = 5
    B = reduction(B, beta, params)

    # if g == 0:
    #     if(g6k.M.get_r(0, 0) <= target_norm):
    #         return g6k.M.B[0]
    #     else:
    #         raise ValueError("No solution found.")

    #print(g6k.M.B, B)


    n = A.ncols
    ell = n - g
    print(n, g, ell)

    print(A)
    print()
    print(Al)
    print()
    print(Ag)

    d = 100 # simhash distance

    #
    # Second part: MiTM
    # BDD Queries
    #

    # sg = bdd_query_plain_hybrid(B, Ag, b, g, n, q, beta, params)
    sg = bdd_query_mitm(B, Ag, b, g, n, q, d, beta, params)

    #print(s)
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
