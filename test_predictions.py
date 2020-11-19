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
from sage.all import *



def gen_small(s, n):
	"""
	s+1 entries of 1s and s entries of -1s
	"""
	deg = n
	coeff_vector = deg*[0]
	coeff_vector[deg-1] = 1
	coeff_vector[0] = 1
	index_set = set({0,deg-1})
	for i in range(s-2):
	# add 1's
		while True:
			index1 = randrange(1,deg-1)
			if not index1 in index_set:
				coeff_vector[index1] = 1
				index_set = index_set.union({index1})
				break
	# add -1's
	for i in range(s):
		while True:
			index2 = randrange(1,deg-1)
			if not index2 in index_set:
				coeff_vector[index2] = -1
				index_set = index_set.union({index2})
				break
	return coeff_vector

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
			B[j, i] = A[i, j]
		B[i+n, i] = q
		B[-1, i] = c[i]
	B[-1, -1] = 1

	B = LLL.reduction(B)
	assert(B[:n] == IntegerMatrix(n, m+1))
	B = B[n:]
	return B

def test_ntru_gsa(n, q):

	Amat = IntegerMatrix.random(n, "uniform", bits=floor(log(q,2)))
	A = matrix(ZZ,[Amat[i] for i in range(Amat.nrows)])
	w = int(n/3.)
	s = vector(ZZ,gen_small(w,n))

	e = vector(ZZ,gen_small(w,n))
	b = A*s + e
	b = vector([b[i]%q for i in range(n)])

	#Basis = primal_lattice_basis(Amat, b, q, n)
	beta, nsamples, prep_rt, GSA = find_beta(n, q, n)
	print('GSA predicted:')
	print(GSA)

	print('beta, nsamples: ', beta, nsamples)
	Basis = primal_lattice_basis(Amat, b, q, nsamples)
	print(Basis)
	g6k = Siever(Basis)

	d = g6k.full_n
	g6k.lll(0, g6k.full_n)
	slope = basis_quality(g6k.M)["/"]
	print("Intial Slope = %.5f\n" % slope)

	print('GSA input:')
	print([g6k.M.get_r(i, i) for i in range(d)])

	print('d:', d)
	target_norm = ceil( (2./3)*d + 1) + 1
	print("target_norm:", target_norm)

	return 1

n = 120
q = 4201
test_ntru_gsa(n,q)
