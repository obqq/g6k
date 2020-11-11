from math import sqrt, pi, e, log, floor, exp, ceil


def get_delta(k):
    """
    Auxiliary function giving root Hermite factors. Small values
    experimentally determined, otherwise from [Chen13]

    :param k: BKZ blocksize for which the root Hermite factor is required

	"""
    small = (( 2, 1.02190),  # noqa
             ( 5, 1.01862),  # noqa
             (10, 1.01616),
             (15, 1.01485),
             (20, 1.01420),
             (25, 1.01342),
             (28, 1.01331),
             (40, 1.01295))

    k = float(k)
    if k <= 2:
        return (1.0219)
    elif k < 40:
        for i in range(1, len(small)):
            if small[i][0] > k:
                return (small[i-1][1])
    elif k == 40:
        return (small[-1][1])
    else:
        return (k/(2*pi*e) * (pi*k)**(1./k))**(1/(2*(k-1.)))

def getGSA(n, q, beta, nrows):
	assert beta!=0
	delta = get_delta(beta)
	d = n+nrows
	log_q = log(q)
	zone1 = (nrows)*[log_q]
	slope = -(1./(beta-1))*log(beta/(2.*pi*e) * (pi*beta)**(1./beta))
	#print(beta/(2.*pi*e) * (pi*beta)**(1./beta))
	#print('slope:', slope)
	zone2_length = int(floor(log_q/-slope))
	zone2 = [log_q + i * slope for i in range(1, zone2_length+1)]
	zone3 = n*[0]
	GSA = zone1+zone2+zone3

	lattice_vol =nrows*log_q
	current_vol = sum(GSA[i] for i in range(d))

	#correct the volume since now current_vol > lattice_vol
	ind = 0
	while current_vol>lattice_vol:
		current_vol -= GSA[ind]
		current_vol += GSA[ind+d]
		ind += 1
	GSA = GSA[ind:ind+d]
	assert ind<=zone2_length

	i_index = max(0, nrows-ind)
	j_index = min(i_index + zone2_length, d)
	#error we make by overshooting the while loop
	err = lattice_vol - current_vol
	for i in range(i_index, j_index):
		GSA[i] += err / (j_index-i_index+1)

	return GSA, i_index, j_index


def getGSA_noq(n, q, beta, nrows):

	assert beta!=0
	d = n+nrows
	delta = get_delta(beta)
	log_delta =log(delta,2)

	log_q = log(q)
	slope = -(1/(beta-1))*log(beta/(2*pi*e)*(pi*beta)**(1/beta))

	lattice_vol = nrows*log_q
	current_vol = 0
	GSA_tmp = 0
	GSA = []
	for i in range(d):
		GSA_tmp -= slope
		current_vol += GSA_tmp
		if lattice_vol<current_vol:
			break
		GSA = [GSA_tmp] + GSA

	zone2_length = len(GSA)
	GSA += (d-zone2_length)*[0]
	i_index = 0
	j_index = min(i_index + zone2_length, d)

	#error we make by overshooting the while loop
	current_vol = sum(GSA[i] for i in range(d))
	err = current_vol - lattice_vol
	for i in range(i_index, j_index):
		GSA[i] -= err / (j_index-i_index+1)

	return GSA, i_index, j_index

def delta_BKZ(b):
	""" The root hermite factor delta of BKZ-b
		"""
	return ((pi*b)**(1./b) * b / (2*pi*exp(1)))**(1./(2.*b-2.))


def construct_BKZ_shape(q, nq, n1, b):
	d = nq+n1
	if b==0:
		L = nq*[log(q)] + n1*[0]
		return (nq, nq, L)

	slope = -2 * log(delta_BKZ(b))
	lq = log(q)
	B = int(floor(log(q) / - slope))    # Number of vectors in the sloppy region
	L = nq*[log(q)] + [lq + i * slope for i in range(1, B+1)] + n1*[0]

	x = 0
	lv = sum (L[:d])
	glv = nq*lq                     # Goal log volume

	while lv > glv:                 # While the current volume exceeeds goal volume, slide the window to the right
		lv -= L[x]
		lv += L[x+d]
		x += 1

	assert x <= B                   # Sanity check that we have not gone too far

	L = L[x:x+d]
	a = max(0, nq - x)             # The length of the [q, ... q] sequence
	B = min(B, d - a)              # The length of the GSA sequence

	diff = glv - lv
	assert abs(diff) < lq               # Sanity check the volume, up to the discretness of index error

	for i in range(a, a+B):        # Small shift of the GSA sequence to equiliBrate volume
		L[i] += diff / B
	lv = sum(L)
	assert abs(lv/glv - 1) < 1e-6        # Sanity check the volume

	return L,a, a + B

def sievig0292(dim):
	return dim*log(sqrt(3./2))/log(2.)

def sievig0349(dim):
	return dim*0.349

def find_beta(n, q, nrows, svp_alg = sievig0292):
	rt_min = 10000 #infinity TODO: try float("inf")
	nsamples_opt = nrows
	beta_opt = 10000 #infinity
	for nsamples in range(nrows/2, nrows, 5):
		dim = n+nsamples
		for beta in range(15, dim,5):
			if svp_alg(beta)>=rt_min:
				break
			#print('find_beta:', n, q, beta, nsamples)
			GSA, i, j = getGSA(n, q, beta, nsamples)
			#print(GSA)
			if exp(GSA[dim-beta]) > sqrt(2.0/3.0 * beta): #2/3 is the error sparsity (assume the error is uniform from {+-1, 0}
				beta_opt = beta
				nsamples_opt = nsamples
				rt_min = svp_alg(beta_opt)
				#print('find_beta', beta_opt, nsamples_opt, rt_min)
				break


	return beta_opt, nsamples_opt, rt_min
