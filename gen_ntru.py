from sage.all import *
import numpy

n = 2**5
deg = n/2
K = CyclotomicField(n)
variable = K.gen()



P = Primes()
q = next_prime(8*n)

#print K, K.degree(), K.polynomial()


def gen_small(B, s):
  coeff_vector = deg*[0]
  coeff_vector[deg-1] = 1
  coeff_vector[0] = 1
  for i in range(s):
    index = ZZ.random_element(1,deg-1)
    value = ZZ.random_element(-B,B)
    coeff_vector[index] = value

  return K(coeff_vector)

F = GF(q)
Fx = PolynomialRing(F, 'x')
Fx_qou = Fx.quotient(K.polynomial(), 'x')
variable_x = Fx_qou.gen()
#print(variable_x)


s = 10
f = Fx_qou(gen_small(1, s+1))
g = Fx_qou(gen_small(1, 10))
h = f/g
#xpoly = Fx_qou(x)




def print_matrix(h, filename):
	n = len(list(h))
	f = file(filename, 'w')
	f.write('[')
	for i in range(n):
		hvector = list(h* variable_x**i)
		#f.write( str(hvector).replace('[','').replace(']','') +'\n')
		f.write( str(hvector)+'\n')
	f.write(']')
	f.close()
	
	return 1
		

#print f
#print g
#print h


filename = 'ntru_n.txt'
print_matrix(h, filename)