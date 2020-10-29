

from sage.all import *



#P = Primes()
#q = next_prime(8*n)

#print K, K.degree(), K.polynomial()


def gen_small(B, s, n):
  deg = n/2
  coeff_vector = deg*[0]
  coeff_vector[deg-1] = 1
  coeff_vector[0] = 1
  for i in range(s):
    index = ZZ.random_element(1,deg-1)
    value = ZZ.random_element(-B,B)
    coeff_vector[index] = value

  return coeff_vector



def print_ntru(q, h, variable_x, filename):
	n = len(list(h))
	f = file(filename, 'w')
	f.write(str(q)+'\n')
	#f.write('[')
	HMat = [0]*n
	for i in range(n):
		hvector = list(h* variable_x**i)
		HMat[i] = hvector
		f.write( str(hvector).replace(',','') +'\n')
		#f.write( str(hvector)+'\n')
	#f.write(']')
	f.close()

	return HMat

def gen_ntru_challenge(n):

	K = CyclotomicField(n)

	P = Primes()
	q = next_prime(11*n)


	F = GF(q)
	Fx = PolynomialRing(F, 'x')
	Fx_qou = Fx.quotient(K.polynomial(), 'x')
	variable_x = Fx_qou.gen()

	sparsity = ceil(n/3.)

	f = Fx_qou(gen_small(1, sparsity+1, n))
	g = Fx_qou(gen_small(1, sparsity, n))
	h = f/g


	filename = 'ntru_n_'+str(n)+'.txt'
	print_ntru(q, h, variable_x, filename)

	return h, q



if __name__ == '__main__':
    gen_ntru_challenge(128)
