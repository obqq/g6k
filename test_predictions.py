from fpylll import GSO, IntegerMatrix, LLL, BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.utils.stats import SieveTreeTracer, dummy_tracer

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from g6k.utils.ntru_hybrid_estimation import plain_hybrid_compleixty, ntru_plain_hybrid_basis
from g6k.utils.ntru_gsa import find_beta

from random import randrange
from math import floor, log, ceil




def primal_lattice_basis(A, c, q, m=None):
	"""
	Construct primal lattice basis for LWE challenge
	``(A,c)`` defined modulo ``q``.

	:param A: LWE matrix
	:param c: LWE vector
	:param q: integer modulus
	:param m: number of samples to use (``None`` means all)

	"""
	if m is None:
		m = A.nrows
	elif m > A.nrows:
		raise ValueError("Only m=%d samples available." % A.nrows)
	n = A.ncols

	B = IntegerMatrix(m+n+1, m+1)
	for i in range(m):
		for j in range(n):
			B[j, i] = A[j, i]
		B[i+n, i] = q
		B[-1, i] = c[i]
	B[-1, -1] = 1
	#print("B:")
	#print(B)
	B = LLL.reduction(B)
	assert(B[:n] == IntegerMatrix(n, m+1))
	B = B[n:]
	return B

def test_ntru_gsa(n):

	# read from file
	filename = 'lwe_n'+str(n)+'.txt'
	data = open(filename, 'r').readlines()
	q = int(data[0])
	B = eval(",".join([s_.replace('\n','').replace(" ", ", ") for s_ in data[1:n+1]]))
	B = IntegerMatrix.from_matrix(B)
	c = eval(data[n+1].replace(" ", ','))

	beta, nsamples, prep_rt, GSA = find_beta(n, q, n)
	print('beta, nsamples:', beta, nsamples)
	print('GSA predicted:')
	print(GSA)

	print('beta, nsamples: ', beta, nsamples)
	Basis = primal_lattice_basis(B, c, q, nsamples)

	g6k = Siever(Basis)

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
	#
	#   Preprocessing
	#
	if beta < 50:
		print("Starting a fpylll BKZ-%d tour. " % (beta))
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

	return 1

def 
n = 128
test_ntru_gsa(n)
