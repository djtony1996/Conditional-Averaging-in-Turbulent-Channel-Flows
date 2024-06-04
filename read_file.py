import numpy as np

def read_bin(filename, dim):
    with open(filename, 'rb') as fid:
        A = np.fromfile(fid, dtype=np.float64, count=np.prod(dim))
        A = np.reshape(A, dim, order='C')
    return A


def get_zp(dz):
    zp = (np.cumsum(np.concatenate(([0], dz[:-1]))) + np.cumsum(dz)) / 2
    return zp


