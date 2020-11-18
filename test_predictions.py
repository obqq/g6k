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
from g6k.utils.ntru_hybrid_estimation import plain_hybrid_compleixty, ntru_plain_hybrid_basis
from g6k.utils.ntru_gsa import find_beta


def test_ntru_gsa(n, q):

	A = IntegerMatrix.random(n, "uniform", bits=floor(log(q)))
	Basis, = ntru_plain_hybrid_basis(A, 0, q, n)
	beta, nsamples, prep_rt, GSA = find_beta(n, q, n)
	print('GSA predicted:')
	print(GSA)

	return 1
