#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:16:47 2024

@author: jitongd
"""

import numpy as np
from cheb_numeric import *
from read_file import *
from derivative_calculation import *
from calculate_TKE_sTKE import *

def get_detection_events(all_events_array, k_scale, detection_case, whe_single):
    if detection_case == 1:
        pick_events = np.abs(all_events_array) > (k_scale * np.sqrt(np.mean(all_events_array**2)))
    elif detection_case == 2:
        pick_events = all_events_array > (k_scale * np.mean(all_events_array))
    elif detection_case == 3:
        pick_events = all_events_array < (k_scale * np.mean(all_events_array))
    elif detection_case == 4:
        pick_events = all_events_array > (k_scale * np.max(all_events_array))
    else:
        raise ValueError("This detection criterion has not been coded.")
    
    # Process for single events
    if whe_single == 1:
        repeat_left = np.zeros(len(pick_events)-1, dtype=int)
        repeat_right = np.zeros(len(pick_events)-1, dtype=int)
        
        # left
        k_array1 = 0
        if pick_events[0]:
            repeat_left[k_array1] = 0
            k_array1 += 1
        
        for k_array in range(len(pick_events)-1):
            if not pick_events[k_array] and pick_events[k_array+1]:
                repeat_left[k_array1] = k_array
                k_array1 += 1
        
        # right
        k_array1 = 0
        for k_array in range(1, len(pick_events)):
            if pick_events[k_array-1] and not pick_events[k_array]:
                repeat_right[k_array1] = k_array 
                k_array1 += 1
        
        if pick_events[-1]:
            repeat_right[k_array1] = len(pick_events) - 1
            k_array1 += 1
        
        repeat_left = repeat_left[repeat_left != 0]
        repeat_right = repeat_right[repeat_right != 0]
        
        repeat_index = np.round((repeat_left + repeat_right) / 2).astype(int) 
        pick_events[:] = 0
        pick_events[repeat_index] = 1

    # Identify positive and negative events
    NonTp_posi = all_events_array > 0
    NonTp_nega = all_events_array < 0
    
    pick_events_posi = NonTp_posi * pick_events
    pick_events_nega = NonTp_nega * pick_events * (-1)

    return pick_events_posi, pick_events_nega



def get_new_3d_box(u, kx, ky, kx_middle, ky_middle):
    # Rearrange along the y-axis
    if ky < ky_middle:
        temp_u = u[:, :-(ky_middle - ky), :]
        u = np.concatenate((u[:, -(ky_middle - ky):, :], temp_u), axis=1)
    else:
        temp_u = u[:, (ky - ky_middle):, :]
        u = np.concatenate((temp_u, u[:, :(ky - ky_middle), :]), axis=1)

    # Rearrange along the x-axis
    if kx < kx_middle:
        temp_u = u[:, :, :-(kx_middle - kx)]
        u = np.concatenate((u[:, :, -(kx_middle - kx):], temp_u), axis=2)
    else:
        temp_u = u[:, :, (kx - kx_middle):]
        u = np.concatenate((temp_u, u[:, :, :(kx - kx_middle)]), axis=2)
    
    return u


def get_swirling_strength(velocity_tensor,nx,ny,nz):
    swirling_strength = np.zeros((nz-1,ny,nx))
    for kz in range(nz-1):
        for ky in range(ny):
            for kx in range(nx):
                velocity_tensor_onepoint = velocity_tensor[kz, ky, kx,:].reshape(3, 3)
                eigenvalues = np.linalg.eigvals(velocity_tensor_onepoint)
                swirling_strength[kz, ky, kx] = np.max(np.imag(eigenvalues))
    return swirling_strength


def get_uvwNonTp_z(k_z,Retau,read_array):
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

    U    = channelRe['Up'][1:-1]
    dUdz = channelRe['Up_diff1'][1:-1]

    Diff, zc = cheb(nz)
    Diff = Diff[1:-1,1:-1]
    
    u_slice = np.zeros((len(read_array),ny,nx))
    v_slice = np.zeros((len(read_array),ny,nx))
    w_slice = np.zeros((len(read_array),ny,nx))
    NonTp_slice = np.zeros((len(read_array),ny,nx))
    Prop_slice  = np.zeros((len(read_array),ny,nx))
    Dissp_slice = np.zeros((len(read_array),ny,nx))
    
    if Retau == 180:
        loadname1 = '180/112x112x150'
    elif Retau == 395:
        loadname1 = '395/256x256x300'
    elif Retau == 590:
        loadname1 = '590/384x384x500'
    else:
        raise ValueError("Unsupported Retau value")

    for k_index in range(len(read_array)):
        u = np.zeros((nzDNS+2,ny,nx))
        v = np.zeros((nzDNS+2,ny,nx))
        w = np.zeros((nzDNS+1,ny,nx))
        
        u[1:-1,:,:] = read_bin('u/u_it{}.dat'.format(read_array[k_index]), np.array([nzDNS,ny,nx]))
        v[1:-1,:,:] = read_bin('v/v_it{}.dat'.format(read_array[k_index]), np.array([nzDNS,ny,nx]))
        w[1:,:,:]   = read_bin('w/w_it{}.dat'.format(read_array[k_index]), np.array([nzDNS,ny,nx]))
        
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/u_it{int(read_array[k_index]):.0f}.dat"
        # u[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/v_it{int(read_array[k_index]):.0f}.dat"
        # v[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/w_it{int(read_array[k_index]):.0f}.dat"
        # w[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        
        u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
        
        u = u - U[:, np.newaxis, np.newaxis]
        Prop, Dissp, NonTp = get_three_energy_physicalspace(u, v, w, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
        
        u_slice[k_index,:,:] = u[k_z,:,:]
        v_slice[k_index,:,:] = v[k_z,:,:]
        w_slice[k_index,:,:] = w[k_z,:,:]
        NonTp_slice[k_index,:,:] = NonTp[k_z,:,:]
        Prop_slice[k_index,:,:]  = Prop[k_z,:,:]
        Dissp_slice[k_index,:,:] = Dissp[k_z,:,:]
        
    u_slice = u_slice.astype(np.float16)
    v_slice = v_slice.astype(np.float16)
    w_slice = w_slice.astype(np.float16)
    NonTp_slice = NonTp_slice.astype(np.float16)
    Prop_slice  = Prop_slice.astype(np.float16)
    Dissp_slice = Dissp_slice.astype(np.float16)
    
    return u_slice, v_slice, w_slice, NonTp_slice, Prop_slice, Dissp_slice



def get_cd_velocities(pick_NonTp_index,kx_detection,ky_detection,kx_middle,ky_middle,Retau,read_array):
    filename = f'full{Retau}_mean.npz'
    data = np.load(filename, allow_pickle=True)
    channelRe = data['channelRe'].item()
    nx = data['nx'].item()
    ny = data['ny'].item()
    nz = data['nz'].item()
    nzDNS = data['nzDNS'].item()
    xu = data['xu']
    xp = data['xp']
    yv = data['yv']
    yp = data['yp']
    zw = data['zw']
    zp = data['zp']
    
    U    = channelRe['Up'][1:-1]
    _, zc = cheb(nz)
    
    u_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    v_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    w_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    
    for k_array_index in range(len(pick_NonTp_index)):
        k_array = pick_NonTp_index[k_array_index]
        
        u = np.zeros((nzDNS+2,ny,nx))
        v = np.zeros((nzDNS+2,ny,nx))
        w = np.zeros((nzDNS+1,ny,nx))
        
        # for local debugging purposes
        u[1:-1,:,:] = read_bin('u/u_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        v[1:-1,:,:] = read_bin('v/v_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        w[1:,:,:]   = read_bin('w/w_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        
        # for HPC code running
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/u_it{int(read_array[k_array]):.0f}.dat"
        # u[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/v_it{int(read_array[k_array]):.0f}.dat"
        # v[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/w_it{int(read_array[k_array]):.0f}.dat"
        # w[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        
        u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
        u = u - U[:, np.newaxis, np.newaxis]
        
        u = get_new_3d_box(u, kx_detection, ky_detection, kx_middle, ky_middle)
        v = get_new_3d_box(v, kx_detection, ky_detection, kx_middle, ky_middle)
        w = get_new_3d_box(w, kx_detection, ky_detection, kx_middle, ky_middle)

        u_cd     = k_array_index*u_cd/(k_array_index+1) + u/(k_array_index+1)
        v_cd     = k_array_index*v_cd/(k_array_index+1) + v/(k_array_index+1)
        w_cd     = k_array_index*w_cd/(k_array_index+1) + w/(k_array_index+1)
    
    return u_cd, v_cd, w_cd


def get_cd_velocities_more(pick_NonTp_index,kx_detection,ky_detection,kx_middle,ky_middle,Retau,read_array):
    filename = f'full{Retau}_mean.npz'
    data = np.load(filename, allow_pickle=True)
    channelRe = data['channelRe'].item()
    nx = data['nx'].item()
    ny = data['ny'].item()
    nz = data['nz'].item()
    nzDNS = data['nzDNS'].item()
    xu = data['xu']
    xp = data['xp']
    yv = data['yv']
    yp = data['yp']
    zw = data['zw']
    zp = data['zp']
    
    U    = channelRe['Up'][1:-1]
    _, zc = cheb(nz)
    
    u_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    v_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    w_cd     = np.zeros((nz-1, ny, nx), dtype=float)
    NonTp_cd = np.zeros((nz-1, ny, nx), dtype=float)
    swirling = np.zeros((nz-1, ny, nx), dtype=float)
    
    for k_array_index in range(len(pick_NonTp_index)):
        k_array = pick_NonTp_index[k_array_index]
        
        u = np.zeros((nzDNS+2,ny,nx))
        v = np.zeros((nzDNS+2,ny,nx))
        w = np.zeros((nzDNS+1,ny,nx))
        
        # for local debugging purposes
        u[1:-1,:,:] = read_bin('u/u_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        v[1:-1,:,:] = read_bin('v/v_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        w[1:,:,:]   = read_bin('w/w_it{}.dat'.format(read_array[k_array]), np.array([nzDNS,ny,nx]))
        
        # for HPC code running
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/u_it{int(read_array[k_array]):.0f}.dat"
        # u[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/v_it{int(read_array[k_array]):.0f}.dat"
        # v[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        # loadname = f"../ChanFast/grid_{loadname1}/outputdir/w_it{int(read_array[k_array]):.0f}.dat"
        # w[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
        
        u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
        u = u - U[:, np.newaxis, np.newaxis]
        
        u = get_new_3d_box(u, kx_detection, ky_detection, kx_middle, ky_middle)
        v = get_new_3d_box(v, kx_detection, ky_detection, kx_middle, ky_middle)
        w = get_new_3d_box(w, kx_detection, ky_detection, kx_middle, ky_middle)
        
        _, _, NonTp = get_three_energy_physicalspace(u, v, w, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
    
        velocity_tensor = get_velocity_tensor(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
        swirling_strength = get_swirling_strength(velocity_tensor,nx,ny,nz)

        u_cd     = k_array_index*u_cd/(k_array_index+1) + u/(k_array_index+1)
        v_cd     = k_array_index*v_cd/(k_array_index+1) + v/(k_array_index+1)
        w_cd     = k_array_index*w_cd/(k_array_index+1) + w/(k_array_index+1)
        NonTp_cd = k_array_index*NonTp_cd/(k_array_index+1) + NonTp/(k_array_index+1)
        swirling = k_array_index*swirling/(k_array_index+1) + swirling_strength/(k_array_index+1)
        
    
    return u_cd, v_cd, w_cd, NonTp_cd, swirling



