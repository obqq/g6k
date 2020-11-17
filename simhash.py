import os
from math import ceil
import numpy as np

XPC_WORD_LEN = 4 # number of 64-bit words of simhashes
XPC_BIT_LEN = 256 # number of bits for simhashes

SIMHASH_CODES_BASEDIR = 'spherical_coding'


class SimHashes:
    '''
    This class stores the hash function we use to compute simhashes and allows to compute a simhash for a given point.
    '''

    def __init__(self, n, sim_hash_rng=None):
        '''
        constructs a SimHashes objects and stores the seed used to set future hash functions.
        Note that reset_compress_pos must be called at least once before compress is used.
        '''


        self.n = n #  Dimension of the entries on which we compute SimHashes.
        self.compress_pos = [] # Indices to chose for compression / SimHashes

        self.sim_hash_rng = sim_hash_rng #  // we use our own rng, seeded during construction.
                    # (This is to make simhash selection less random in multithreading and to actually simplify some internal code)

        self.reset_compress_pos()
        print(self.compress_pos)

    def reset_compress_pos(self):
        '''
        Recomputes the sparse vector defining the current compression function / Simhash.
        This is called during changes of context / basis switches.
        Note that this makes a recomputation of all the simhashes stored in db / cdb neccessary.
        '''
        if self.n < 30:
            self.compress_pos = [[0]*6 for i in range(XPC_BIT_LEN)]
            return

        file_path = os.path.join(SIMHASH_CODES_BASEDIR, f'sc_{self.n}_{XPC_BIT_LEN}.def')
        if not os.path.exists(file_path):
            raise ValueError(f'File {file_path} not found!!')

        file = open(file_path, 'r')

        if self.sim_hash_rng:
            np.random.seed(seed)

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
        Since it is a function of the normalized GSO coos, we pass the yr - coos.
        '''

        # Compute the compressed representation of an entry
        # compress(std::array<LFT,MAX_SIEVING_DIM> const &v) const

        # ATOMIC_CPUCOUNT(260);
        c = [0]
        if n < 30:
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

                c_tmp <<= 1
                if a > 0:
                    c_tmp |= a
            c[j] = c_tmp
        return c


def binary_search(v, t, v_len):
    '''
    O(2 log n)
    '''
    pos1, pos2 = None, None

    l = 0
    r = v_len - 1

    while r > l:
        m = ceil((l + r) / 2)
        print(l, m, r)

        if v[m] == t:
            l = m
            break

        if v[m] >= t:
            r = m - 1
        else:
            l = m + 1

    l_pos = l

    l = 0
    r = v_len - 1

    while r > l:
        m = ceil((l + r) / 2)
        print(l, m, r)

        if v[m] == t:
            l = m
            break

        if v[m] > t:
            r = m - 1
        else:
            l = m + 1

    r_pos = l

    return l_pos, r_pos


def search(V1, v_hash, d):
    '''
    '''
    V1_slice = V1
    V1_len = len(V1)
    for i in range(XPC_WORD_LEN - 1):
        row1, row2 = binary_search(V1_slice[:, i], v_hash[i], V1_len)

        if not row1 and not row2:
            return
        V1_slice = V1_slice[row1:row2, :]
        V1_len = row - row1

        if w[i] != v[i]:
            continue

    # search closest

    return


def closest_pairs(V1, V2, n, step=1):
    '''
    Searches for close pairs in sets V1 and V2
    '''
    SH = SimHashes(n)

    V1 = np.array(np.array([v] + SH.compress(v)) for v in V1)
    V1 = sorted(V1, key=lambda x: x[1])

    for v in V2:
        v_hash = SH.compress(v)

        close_vecs = search(V1, v_hash, d, len_V1)
        if close_vecs:
            return (v, close_vecs)


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
    V = [0, 1, 2, 4, 5, 5, 5, 5, 6, 7, 7, 9]
    n = len(V)
    print(binary_search(V, 5, n))


def main():
    test3()


if __name__ == '__main__':
    main()
