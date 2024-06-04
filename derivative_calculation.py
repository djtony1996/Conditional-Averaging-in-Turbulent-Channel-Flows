#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:51:17 2024

@author: jitongd
"""

import numpy as np

def getkm(alpha_array, beta_array, nx, ny, dkx, dky):
    kx_m = np.zeros((ny, nx))
    ky_m = np.zeros((ny, nx))
    
    for k_array in range(len(alpha_array)):
        index_kx = round(alpha_array[k_array] / dkx) 
        index_ky = round(beta_array[k_array] / dky) 
        
        if alpha_array[k_array] < 0:
            kx_m[index_ky, nx + index_kx] = alpha_array[k_array]
            ky_m[index_ky, nx + index_kx] = beta_array[k_array]
        else:
            kx_m[index_ky, index_kx] = alpha_array[k_array]
            ky_m[index_ky, index_kx] = beta_array[k_array]
    
    kx_m[ny-1:ny//2+1:-1, :] = kx_m[1:ny//2-1, :]
    ky_m[ny-1:ny//2+1:-1, :] = -ky_m[1:ny//2-1, :]
    
    return kx_m, ky_m

# Example usage
# alpha_array = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, -3, -3, -3, -2,-2 ,-2, -1, -1, -1])
# beta_array = np.array([2, 4, 6, 2, 4, 6,2, 4, 6, 2,4 ,6, 2, 4, 6, 2, 4, 6])
# nx = 16
# ny = 16
# dkx = 1
# dky = 2

# kx_m, ky_m = getkm(alpha_array, beta_array, nx, ny, dkx, dky)


def get_3d(f, Diff, kx_m, ky_m):
    f = np.transpose(f, (1, 2, 0))
    dfdx = 1j * kx_m * f
    dfdy = 1j * ky_m * f
    
    f = np.transpose(f, (2, 0, 1))
    dfdx = np.transpose(dfdx, (2, 0, 1))
    dfdy = np.transpose(dfdy, (2, 0, 1))
    
    dfdz = np.einsum('ijk,ikl->ijl', Diff, f)
    
    return dfdx, dfdy, dfdz

# Example usage
f = np.random.rand(3, 4, 5)
Diff = np.random.rand(5, 5, 4)
kx_m = np.random.rand(4, 5, 3)
ky_m = np.random.rand(4, 5, 3)

dfdx, dfdy, dfdz = get_3d(f, Diff, kx_m, ky_m)
print("dfdx:", dfdx)
print("dfdy:", dfdy)
print("dfdz:", dfdz)

