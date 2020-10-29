#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NTRU
"""

from __future__ import absolute_import
from __future__ import print_function
import copy
import re
import sys
import time

import os.path

from collections import OrderedDict # noqa
from math import log

from fpylll import IntegerMatrix, BKZ as fplll_bkz
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

from g6k.utils.ntru_hybrid_estimation import plain_hybrid_compleixty, ntru_plain_hybrid_basis


def read_ntru_from_file(filename):
    if not os.path.isfile(filename):
        raise ValueError ('file ', filename, 'is not found')
    data = open(filename, "r").readlines()
    q = int(data[0])
    H = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[1 :]]))
    H = IntegerMatrix.from_matrix(H)
    return H, q


def ntru_kernel(arg0, params=None, seed=None):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/alpha: the noise rate of the LWE-challenge

        - lwe/m: the number of samples to use for the primal attack

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

    # flow of the lwe solver
    svp_bkz_time_factor = params.pop("lwe/svp_bkz_time_factor")
    goal_margin = params.pop("lwe/goal_margin")

    # generation of lwe instance and Kannan's embedding
    alpha = params.pop("lwe/alpha")
    m = params.pop("lwe/m")
    decouple = svp_bkz_time_factor > 0

    # misc
    dont_trace = params.pop("dummy_tracer")
    verbose = params.pop("verbose")

    filename = 'ntru_n_'+str(n)+'.txt'
    H, q = read_ntru_from_file(filename)

    print("-------------------------")
    print("Hybrid attack on NTRU n=%d" %n)


    # compute the attack parameters
    paramset_NTRU1 = {'n': n, 'q': q, 'w': 2*(n/3)}
    print(paramset_NTRU1)
    beta, g, rt = plain_hybrid_compleixty(paramset_NTRU1, verbose = True)

    B = ntru_plain_hybrid_basis(A, g, q, m=m)

    g6k = Siever(B, params)
    print("GSO precision: ", g6k.M.float_type)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("ntru"), start_clocks=True)

    d = g6k.full_n
    g6k.lll(0, g6k.full_n)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    T0 = time.time()
    T0_BKZ = time.time()
    for blocksize in blocksizes:
        for tt in range(tours):
            # BKZ tours

            if blocksize < fpylll_crossover:
                if verbose:
                    print("Starting a fpylll BKZ-%d tour. " % (blocksize), end=' ')
                    sys.stdout.flush()
                bkz = BKZReduction(g6k.M)
                par = fplll_bkz.Param(blocksize,
                                      strategies=fplll_bkz.DEFAULT_STRATEGY,
                                      max_loops=1)
                bkz(par)

            else:
                if verbose:
                    print("Starting a pnjBKZ-%d tour. " % (blocksize))

                pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                     verbose=verbose,
                                     extra_dim4free=extra_dim4free,
                                     dim4free_fun=dim4free_fun,
                                     goal_r0=target_norm,
                                     pump_params=pump_params)

            T_BKZ = time.time() - T0_BKZ

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))

            g6k.lll(0, g6k.full_n)

            if g6k.M.get_r(0, 0) <= target_norm:
                break

            # overdoing n_max would allocate too much memory, so we are careful
            svp_Tmax = svp_bkz_time_factor * T_BKZ
            n_max = int(58 + 2.85 * log(svp_Tmax * params.threads)/log(2.))

            rr = [g6k.M.get_r(i, i) for i in range(d)]
            for n_expected in range(2, d-2):
                x = (target_norm/goal_margin) * n_expected/(1.*d)
                if 4./3 * gaussian_heuristic(rr[d-n_expected:]) > x:
                    break

            print("Without otf, would expect solution at pump-%d. n_max=%d in the given time." % (n_expected, n_max)) # noqa
            if n_expected >= n_max - 1:
                continue

            n_max += 1

            # Larger SVP

            llb = d - blocksize
            while gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)]) < target_norm * (d - llb)/(1.*d): # noqa
                llb -= 1

            f = d-llb-n_max
            if verbose:
                print("Starting svp pump_{%d, %d, %d}, n_max = %d, Tmax= %.2f sec" % (llb, d-llb, f, n_max, svp_Tmax)) # noqa
            pump(g6k, tracer, llb, d-llb, f, verbose=verbose,
                 goal_r0=target_norm * (d - llb)/(1.*d))

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "\n slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))
                print()

            g6k.lll(0, g6k.full_n)
            T0_BKZ = time.time()
            if g6k.M.get_r(0, 0) <= target_norm:
                break

        if g6k.M.get_r(0, 0) <= target_norm:
            print("Finished! TT=%.2f sec" % (time.time() - T0))
            print(g6k.M.B[0])
            alpha_ = int(alpha*1000)
            filename = 'lwechallenge/%03d-%03d-solution.txt' % (n, alpha_)
            fn = open(filename, "w")
            fn.write(str(g6k.M.B[0]))
            fn.close()
            return
	"""
    raise ValueError("No solution found.")

def ntru():
    """
    Attempt to solve an ntru instance.

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


if __name__ == '__main__':
    ntru()
