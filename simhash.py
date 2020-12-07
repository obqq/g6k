import os
from math import inf, ceil
import numpy as np

XPC_WORD_LEN = 4 # number of 64-bit words of simhashes
XPC_BIT_LEN = 256 # number of bits for simhashes

SIMHASH_CODES_BASEDIR = 'spherical_coding'

DEBUG = False

class SimHashes:
    '''
    This class stores the hash function we use to compute simhashes and allows to compute a simhash for a given point.
    '''

    def __init__(self, n, seed=None):
        '''
        constructs a SimHashes objects and stores the seed used to set future hash functions.
        Note that reset_compress_pos must be called at least once before compress is used.
        '''


        self.n = n #  Dimension of the entries on which we compute SimHashes.
        self.compress_pos = [] # Indices to chose for compression / SimHashes

        self.seed = seed

        # self.sim_hash_rng = sim_hash_rng #  // we use our own rng, seeded during construction.
        #             # (This is to make simhash selection less random in multithreading and to actually simplify some internal code)

        self.reset_compress_pos()

    def reset_compress_pos(self):
        '''
        Recomputes the sparse vector defining the current compression function / Simhash.
        This is called during changes of context / basis switches.
        Note that this makes a recomputation of all the simhashes stored in db / cdb neccessary.
        '''
        if not DEBUG and self.n < 30:
            self.compress_pos = [[0]*6 for i in range(XPC_BIT_LEN)]
            return

        file_path = os.path.join(SIMHASH_CODES_BASEDIR, f'sc_{self.n}_{XPC_BIT_LEN}.def')
        if not os.path.exists(file_path):
            raise ValueError(f'File {file_path} not found!!')

        file = open(file_path, 'r')

        if self.seed:
            np.random.seed(self.seed)

        # Create random permutation of {0,..,n-1}
        permut = list(range(0, self.n))
        np.random.shuffle(permut)

        for i in range(0, XPC_BIT_LEN):
            v = file.readline().split(' ')
            if not v:
                print(f'File ended before XPC_BIT_LEN={XPC_BIT_LEN} iteration')
                break
            i_line = []
            for j in range(0, 6):
                k = int(v[j])

                i_line.append(permut[k])
            self.compress_pos.append(i_line)

    def compress(self, v):
        '''
        Compute the compressed representation of an entry.
        '''

        c = []
        if not DEBUG and self.n < 30:
            return c

        for j in range(XPC_WORD_LEN):
            c_tmp = 0
            a = 0
            for i in range(64):
                k = 64 * j + i
                a = v[self.compress_pos[k][0]]
                a += v[self.compress_pos[k][1]]
                a += v[self.compress_pos[k][2]]
                a -= v[self.compress_pos[k][3]]
                a -= v[self.compress_pos[k][4]]
                a -= v[self.compress_pos[k][5]]

                c_tmp >>= 1 # todo
                if a > 0:
                    c_tmp |= a

            c.append(c_tmp) # todo: % self.n   ?
        return c


def search_range(v, t, v_len):
    '''
    O(2 log n)
    https://stackoverflow.com/questions/30794533/how-to-do-a-binary-search-for-a-range-of-the-same-value
    '''
    if t < v[0] or t > v[-1]:
        return None, None

    r = 0
    h = v_len
    while r < h:
        m = (r + h) // 2
        if t < v[m]:
            h = m
        else:
            r = m + 1

    l = 0
    h = r - 1
    while l < h:
        m = (l + h) // 2
        if t > v[m]:
            l = m + 1
        else:
            h = m

    # todo: optimize for cases when t not in v
    if t not in v[l:r]:
        return None, None

    return l, r


def search(V1, v_hash, d):
    '''
    '''
    V1_slice = V1
    V1_len = len(V1)
    for i in range(XPC_WORD_LEN - 1):
        row1, row2 = search_range(V1_slice[:, i], v_hash[i], V1_len)

        if not row1 and not row2:
            return

        V1_slice = V1_slice[row1:row2, :]
        V1_len = row2 - row1

        # if w[i] != v[i]:
        #     continue

    # search closest

    closest_w, miv_v_dist = None, inf
    for w in V1_slice:
        dist = abs(v_hash[XPC_WORD_LEN - 1] - w[XPC_WORD_LEN - 1])
        if dist < d and dist < miv_v_dist:
            closest_w, miv_v_dist = w, dist

    return closest_w


def closest_pairs(V1, V2, n, q, d):
    '''
    Searches for close pairs in sets V1 and V2
    '''
    SH = SimHashes(n)

    V1 = np.array([np.array(SH.compress(v) + v) for v in V1])

    # sorting by multiple columns tests:
    # https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    V1 = V1[np.lexsort([V1[:,i] for i in range(XPC_WORD_LEN, -1, -1)])]

    # print(V1[:,0:XPC_WORD_LEN])

    for v in V2:
        v_hash = SH.compress(v)

        close_vec = search(V1, v_hash, d)
        if close_vec is not None: # todo
            print((v, v_hash), (close_vec[XPC_WORD_LEN:], close_vec[:XPC_WORD_LEN]))
            return close_vec[XPC_WORD_LEN:]


def test1():
    n = 25
    print(f'n={n}')
    SH = SimHashes(n)

    n = 30
    print(f'n={n}')
    SH = SimHashes(n)


def test2():
    SH = SimHashes(3)
    v1 = []


def test3():
    V = [1, 1, 2, 4, 4, 5, 6, 6, 7, 7, 9]
    n = len(V)
    print(search_range(V, 3, n))


def test4():
    n = 7
    V1 = np.array([np.array([np.random.randint(n) for _ in range(n)]) for _ in range(10)])
    print(V1)
    V1 = V1[np.lexsort([V1[:,i] for i in range(XPC_WORD_LEN, -1, -1)])]
    print(V1)

def test5():
    from fpylll import CVP

    # todo: tests with builtin CVP

    global DEBUG
    DEBUG = True

    n = 7
    # n = 31
    d = 10

    np.random.seed(1337)

    V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(100)]
    V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(100)]
    print(closest_pairs(V1, V2, n, d))

    V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(1000)]
    V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(1000)]
    print(closest_pairs(V1, V2, n, d))

    V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(10000)]
    V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(10000)]
    print(closest_pairs(V1, V2, n, d))

    # np.random.seed(1336)
    #
    # V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(100)]
    # V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(100)]
    # print(closest_pairs(V1, V2, n, d))
    #
    # V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(1000)]
    # V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(1000)]
    # print(closest_pairs(V1, V2, n, d))
    #
    # V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(10000)]
    # V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(10000)]
    # print(closest_pairs(V1, V2, n, d))
    #
    # V1 = [[np.random.randint(n) for _ in range(n)] for _ in range(100000)]
    # V2 = [[np.random.randint(n) for _ in range(n)] for _ in range(100000)]
    # print(closest_pairs(V1, V2, n, d))
    #





def main():
    test5()


if __name__ == '__main__':
    main()
