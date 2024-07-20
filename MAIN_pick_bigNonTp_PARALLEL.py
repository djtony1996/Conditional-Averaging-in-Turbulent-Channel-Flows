#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:19:32 2024

@author: jitongd
"""

import sys
import numpy as np
from cheb_numeric import *
from read_file import *
from derivative_calculation import *
from calculate_TKE_sTKE import *
from conditional_averaging import *
import multiprocessing as mp


def get_cd_velocities_posi_nega(kx_detection, ky_detection, NonTp_slice,k_scale,kx_middle,ky_middle,Retau,read_array):
    print("starting")
    NonTp_slice2 = np.squeeze(NonTp_slice[:,ky_detection,kx_detection])
    pick_NonTp_posi, pick_NonTp_nega = get_detection_events(NonTp_slice2, k_scale, 1, 0)
    
    pick_NonTp_posi_index = np.where(pick_NonTp_posi)[0]
    pick_NonTp_nega_index = np.where(pick_NonTp_nega)[0]
        
    u_cd_posi, v_cd_posi, w_cd_posi = get_cd_velocities(pick_NonTp_posi_index,kx_detection,ky_detection,kx_middle,ky_middle,Retau,read_array)
    u_cd_nega, v_cd_nega, w_cd_nega = get_cd_velocities(pick_NonTp_nega_index,kx_detection,ky_detection,kx_middle,ky_middle,Retau,read_array)
    
    u_cd_posi = u_cd_posi.astype(np.float16)
    v_cd_posi = v_cd_posi.astype(np.float16)
    w_cd_posi = w_cd_posi.astype(np.float16)
    u_cd_nega = u_cd_nega.astype(np.float16)
    v_cd_nega = v_cd_nega.astype(np.float16)
    w_cd_nega = w_cd_nega.astype(np.float16)
    
    return len(pick_NonTp_posi_index), len(pick_NonTp_nega_index), u_cd_posi, v_cd_posi, w_cd_posi, u_cd_nega, v_cd_nega, w_cd_nega

if __name__ == '__main__':
    # stating the arguments directly
    Retau = 180
    k_z = 110
    k_scale = 0
    kx_detection_array = np.arange(0,112,120)
    ky_detection_array = np.arange(0,112,20)
    read_array = [60000,200000] 
    jobid = 1224
    workers = 2
    
    
    # for command line arguments passing
    # Retau               = int(sys.argv[1])
    # k_z                 = int(sys.argv[2])
    # k_scale             = int(sys.argv[3])
    # detection_interval  = int(sys.argv[4])
    # read_array_start    = int(sys.argv[5])
    # read_array_end      = int(sys.argv[6])
    # read_array_interval = int(sys.argv[7])
    # jobid               = int(sys.argv[8])
    # workers             = int(sys.argv[9])
    # read_array = np.arange(read_array_start,read_array_end+1,read_array_interval)
    
    if Retau == 180:
        loadname1 = '180/112x112x150'
        nx_d = 50
        ny_d = 50
    elif Retau == 395:
        loadname1 = '395/256x256x300'
    elif Retau == 590:
        loadname1 = '590/384x384x500'
        nx_d = 180
        ny_d = 180
    else:
        raise ValueError("Unsupported Retau value")
        
        
    filename = f'full{Retau}_mean.npz'
    data = np.load(filename, allow_pickle=True)
    channelRe = data['channelRe'].item()
    nx = data['nx'].item()
    ny = data['ny'].item()
    nz = data['nz'].item()
    nzDNS = data['nzDNS'].item()
    dkx = data['dkx'].item()
    dky = data['dky'].item()
    nx_d = data['nx_d'].item()
    ny_d = data['ny_d'].item()
    xu = data['xu']
    xp = data['xp']
    yv = data['yv']
    yp = data['yp']
    zw = data['zw']
    zp = data['zp']
    
    #kx_detection_array = np.arange(0,nx,detection_interval)
    #ky_detection_array = np.arange(0,ny,detection_interval)
    kx_middle = int(nx/2)
    ky_middle = int(ny/2)
    
    U    = channelRe['Up'][1:-1]
    dUdz = channelRe['Up_diff1'][1:-1]
    
    Diff, zc = cheb(nz)
    Diff = Diff[1:-1,1:-1]
    
    _, _, _, NonTp_slice, _, _ = get_uvwNonTp_z(k_z,Retau, read_array)
    print("all good now")
    
    pool = mp.Pool(workers)
    results = pool.starmap(get_cd_velocities_posi_nega, [(kx_detection, ky_detection, NonTp_slice,k_scale,kx_middle,ky_middle,Retau,read_array) for kx_detection in kx_detection_array for ky_detection in ky_detection_array])
    pool.close()
    
#%%
    u_cd_posi_all     = u_cd_posi_all     / np.sum(number_posi)
    v_cd_posi_all     = v_cd_posi_all     / np.sum(number_posi)
    w_cd_posi_all     = w_cd_posi_all     / np.sum(number_posi)
    u_cd_nega_all     = u_cd_nega_all     / np.sum(number_nega)
    v_cd_nega_all     = v_cd_nega_all     / np.sum(number_nega)
    w_cd_nega_all     = w_cd_nega_all     / np.sum(number_nega)
    
    
    _, _, NonTp_posi_aftercd = get_three_energy_physicalspace(u_cd_posi_all, v_cd_posi_all, w_cd_posi_all, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
    _, _, NonTp_nega_aftercd = get_three_energy_physicalspace(u_cd_nega_all, v_cd_nega_all, w_cd_nega_all, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
    
    velocity_tensor       = get_velocity_tensor(u_cd_posi_all, v_cd_posi_all, w_cd_posi_all, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    swirling_posi_aftercd = get_swirling_strength(velocity_tensor,nx,ny,nz)
    velocity_tensor       = get_velocity_tensor(u_cd_nega_all, v_cd_nega_all, w_cd_nega_all, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    swirling_nega_aftercd = get_swirling_strength(velocity_tensor,nx,ny,nz)
    
    u_cd_posi_all         = u_cd_posi_all.astype(np.half)
    v_cd_posi_all         = v_cd_posi_all.astype(np.half)
    w_cd_posi_all         = w_cd_posi_all.astype(np.half)
    NonTp_posi_aftercd    = NonTp_posi_aftercd.astype(np.half)
    swirling_posi_aftercd = swirling_posi_aftercd.astype(np.half)
    u_cd_nega_all         = u_cd_nega_all.astype(np.half)
    v_cd_nega_all         = v_cd_nega_all.astype(np.half)
    w_cd_nega_all         = w_cd_nega_all.astype(np.half)
    NonTp_nega_aftercd    = NonTp_nega_aftercd.astype(np.half)
    swirling_nega_aftercd = swirling_nega_aftercd.astype(np.half)
    
    savename = f'store_files/{Retau}_cd_NonTp_{k_z}_{k_scale}_{jobid}.npz'
    np.savez(savename,
             u_cd_posi_all=u_cd_posi_all,
             v_cd_posi_all=v_cd_posi_all,
             w_cd_posi_all=w_cd_posi_all,
             NonTp_posi_aftercd=NonTp_posi_aftercd,
             swirling_posi_aftercd=swirling_posi_aftercd,
             u_cd_nega_all=u_cd_nega_all,
             v_cd_nega_all=v_cd_nega_all,
             w_cd_nega_all=w_cd_nega_all,
             NonTp_nega_aftercd=NonTp_nega_aftercd,
             swirling_nega_aftercd=swirling_nega_aftercd)
