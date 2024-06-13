#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:16:06 2024

@author: jitongd
"""

import numpy as np

k_z = 110
k_scale = 3
kx_detection_array = 
ky_detection_array = 
Retau = 
read_array = 
jobid = 
workers

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

kx_middle = nx/2
ky_middle = ny/2

U    = channelRe['Up'][1:-1]
dUdz = channelRe['Up_diff1'][1:-1]

Diff, zc = cheb(nz)
Diff = Diff[1:-1,1:-1]

_, _, _, NonTp_slice, _, _ = get_uvwNonTp_z(k_z,Retau, read_array)
print("all good now")

u_cd_posi_all     = np.zeros((nz-1, ny, nx), dtype=float)
v_cd_posi_all     = np.zeros((nz-1, ny, nx), dtype=float)
w_cd_posi_all     = np.zeros((nz-1, ny, nx), dtype=float)
uw_cd_posi_all    = np.zeros((nz-1, ny, nx), dtype=float)
NonTp_cd_posi_all = np.zeros((nz-1, ny, nx), dtype=float)
swirling_posi_all = np.zeros((nz-1, ny, nx), dtype=float)
u_cd_nega_all     = np.zeros((nz-1, ny, nx), dtype=float)
v_cd_nega_all     = np.zeros((nz-1, ny, nx), dtype=float)
w_cd_nega_all     = np.zeros((nz-1, ny, nx), dtype=float)
uw_cd_nega_all    = np.zeros((nz-1, ny, nx), dtype=float)
NonTp_cd_nega_all = np.zeros((nz-1, ny, nx), dtype=float)
swirling_nega_all = np.zeros((nz-1, ny, nx), dtype=float)

for kx_detection_index in range(len(kx_detection_array)):
    print(kx_detection_index)
    kx_detection = kx_detection_array[kx_detection_index]
    
    for ky_detection_index in range(len(ky_detection_array)):
        ky_detection = ky_detection_array(ky_detection_index)
        
        NonTp_slice2 = np.squeeze(NonTp_slice[:,ky_detection,kx_detection])
        pick_NonTp_posi, pick_NonTp_nega = get_detection_events(NonTp_slice2, k_scale, 1, 0)
        
        pick_NonTp_posi_index = np.where(pick_NonTp_posi)[0]
        pick_NonTp_nega_index = np.where(pick_NonTp_nega)[0]
        
        number_posi[ky_detection_index,kx_detection_index] = len(pick_NonTp_posi_index)
        number_nega[ky_detection_index,kx_detection_index] = len(pick_NonTp_nega_index)
        
        u_cd_posi = np.zeros((nz-1, ny, nx), dtype=float)
        v_cd_posi = np.zeros((nz-1, ny, nx), dtype=float)
        w_cd_posi = np.zeros((nz-1, ny, nx), dtype=float)
        uw_cd_posi = np.zeros((nz-1, ny, nx), dtype=float)
        NonTp_cd_posi = np.zeros((nz-1, ny, nx), dtype=float)
        swirling_posi = np.zeros((nz-1, ny, nx), dtype=float)
        u_cd_nega = np.zeros((nz-1, ny, nx), dtype=float)
        v_cd_nega = np.zeros((nz-1, ny, nx), dtype=float)
        w_cd_nega = np.zeros((nz-1, ny, nx), dtype=float)
        uw_cd_nega = np.zeros((nz-1, ny, nx), dtype=float)
        NonTp_cd_nega = np.zeros((nz-1, ny, nx), dtype=float)
        swirling_nega = np.zeros((nz-1, ny, nx), dtype=float)
        
        for k_array_index in range(len(pick_NonTp_posi_index)):
            k_array = pick_NonTp_posi_index(k_array_index)
            
            u = np.zeros((nzDNS+2,ny,nx))
            v = np.zeros((nzDNS+2,ny,nx))
            w = np.zeros((nzDNS+1,ny,nx))
            
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/u_it{int(read_array[k_array]):.0f}.dat"
            u[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/v_it{int(read_array[k_array]):.0f}.dat"
            v[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/w_it{int(read_array[k_array]):.0f}.dat"
            w[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            
            u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
            u = u - U[:, np.newaxis, np.newaxis]
            
            u = get_new_3d_box(u, kx_detection, ky_detection, kx_middle, ky_middle)
            v = get_new_3d_box(v, kx_detection, ky_detection, kx_middle, ky_middle)
            w = get_new_3d_box(w, kx_detection, ky_detection, kx_middle, ky_middle)
        
            _, _, NonTp = get_three_energy_physicalspace(u, v, w, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
        
            velocity_tensor = get_velocity_tensor(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
            swirling_strength = get_swirling_strength(velocity_tensor,nx,ny,nz)

            u_cd_posi = (k_array_index-1).*u_cd_posi./k_array_index + u./k_array_index
            v_cd_posi = (k_array_index-1).*v_cd_posi./k_array_index + v./k_array_index
            w_cd_posi = (k_array_index-1).*w_cd_posi./k_array_index + w./k_array_index
            uw_cd_posi= (k_array_index-1).*uw_cd_posi./k_array_index + (u.*w)./k_array_index
            NonTp_cd_posi = (k_array_index-1).*NonTp_cd_posi./k_array_index + NonTp./k_array_index
            swirling_posi = (k_array_index-1).*swirling_posi./k_array_index + (swirling_strength)./k_array_index
            
        u_cd_posi_all        = u_cd_posi * len(pick_NonTp_posi_index)        + u_cd_posi_all
        v_cd_posi_all        = v_cd_posi * len(pick_NonTp_posi_index)        + v_cd_posi_all
        w_cd_posi_all        = w_cd_posi * len(pick_NonTp_posi_index)        + w_cd_posi_all
        uw_cd_posi_all       = uw_cd_posi * len(pick_NonTp_posi_index)       + uw_cd_posi_all
        NonTp_cd_posi_all    = NonTp_cd_posi * len(pick_NonTp_posi_index)    + NonTp_cd_posi_all
        swirling_cd_posi_all = swirling_cd_posi * len(pick_NonTp_posi_index) + swirling_cd_posi_all
        
        
        for k_array_index in range(len(pick_NonTp_nega_index)):
            k_array = pick_NonTp_nega_index(k_array_index)
            
            u = np.zeros((nzDNS+2,ny,nx))
            v = np.zeros((nzDNS+2,ny,nx))
            w = np.zeros((nzDNS+1,ny,nx))
            
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/u_it{int(read_array[k_array]):.0f}.dat"
            u[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/v_it{int(read_array[k_array]):.0f}.dat"
            v[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            loadname = f"../ChanFast/grid_{loadname1}/outputdir/w_it{int(read_array[k_array]):.0f}.dat"
            w[1:-1,:,:] = read_bin(loadname, np.array([nzDNS,ny,nx]))
            
            u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
            u = u - U[:, np.newaxis, np.newaxis]
            
            u = get_new_3d_box(u, kx_detection, ky_detection, kx_middle, ky_middle)
            v = get_new_3d_box(v, kx_detection, ky_detection, kx_middle, ky_middle)
            w = get_new_3d_box(w, kx_detection, ky_detection, kx_middle, ky_middle)
        
            _, _, NonTp = get_three_energy_physicalspace(u, v, w, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz)
        
            velocity_tensor = get_velocity_tensor(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
            swirling_strength = get_swirling_strength(velocity_tensor,nx,ny,nz)

            u_cd_nega = (k_array_index-1).*u_cd_nega./k_array_index + u./k_array_index
            v_cd_nega = (k_array_index-1).*v_cd_nega./k_array_index + v./k_array_index
            w_cd_nega = (k_array_index-1).*w_cd_nega./k_array_index + w./k_array_index
            uw_cd_nega= (k_array_index-1).*uw_cd_nega./k_array_index + (u.*w)./k_array_index
            NonTp_cd_nega = (k_array_index-1).*NonTp_cd_nega./k_array_index + NonTp./k_array_index
            swirling_nega = (k_array_index-1).*swirling_nega./k_array_index + (swirling_strength)./k_array_index
            
        u_cd_nega_all        = u_cd_nega * len(pick_NonTp_nega_index)        + u_cd_nega_all
        v_cd_nega_all        = v_cd_nega * len(pick_NonTp_nega_index)        + v_cd_nega_all
        w_cd_nega_all        = w_cd_nega * len(pick_NonTp_nega_index)        + w_cd_nega_all
        uw_cd_nega_all       = uw_cd_nega * len(pick_NonTp_nega_index)       + uw_cd_nega_all
        NonTp_cd_nega_all    = NonTp_cd_nega * len(pick_NonTp_nega_index)    + NonTp_cd_nega_all
        swirling_cd_nega_all = swirling_cd_nega * len(pick_NonTp_nega_index) + swirling_cd_nega_all
