import itertools
import scipy.special
import copy

def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        print(bits)
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(vector(GF(2),s))
    return result

def multinom(n, c):

    assert sum(c) == n, 'bad input to multinom!'
    res = 1
    n_ = n
    for i in range(len(c)):
        res*=scipy.special.binom(n_, c[i])
        n_ = n_ - c[i]
    return res

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

def kbits2(n, k):
    """
        k position of "1"s
        k position of "-1"s
    """
    assert 2*k<=n
    full_range = range(n)
    # create all combinations of length k of indicies {0,...n-1} for indexing "1"s
    for pos1 in combinations(full_range, k):
        # remove pos1 from the range of pos2
        # create all combinations of length k of indicies {0,...n-1} for indexing "-1"s
        new_range = []
        start_pos = 0
        for puncture in pos1:
            new_range+=tuple(range(start_pos, puncture))
            start_pos = puncture+1
        print(pos1, new_range)
        #for pos2 in combinations(n,k):
            #check if they interstec
        #   return 1


print(kbits2(4,2))
