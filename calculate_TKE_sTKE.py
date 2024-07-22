#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:56:31 2024

@author: jitongd
"""

import numpy as np
from derivative_calculation import get_3d_physical

def get_three_energy_physicalspace(u, v, w, nx_d, ny_d, nx, ny, dkx, dky, Diff, dUdz):
    
    du_dx, du_dy, du_dz = get_3d_physical(u, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dv_dx, dv_dy, dv_dz = get_3d_physical(v, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dw_dx, dw_dy, dw_dz = get_3d_physical(w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    
    Prop = -u * w * dUdz[:, np.newaxis, np.newaxis]

    Dissxp = -du_dx * du_dx - du_dy * du_dy - du_dz * du_dz
    Dissyp = -dv_dx * dv_dx - dv_dy * dv_dy - dv_dz * dv_dz
    Disszp = -dw_dx * dw_dx - dw_dy * dw_dy - dw_dz * dw_dz
    Dissp  = Dissxp + Dissyp + Disszp

    NonTxp = -u * u * du_dx - u * v * du_dy - u * w * du_dz
    NonTyp = -v * u * dv_dx - v * v * dv_dy - v * w * dv_dz
    NonTzp = -w * u * dw_dx - w * v * dw_dy - w * w * dw_dz
    NonTp  = NonTxp + NonTyp + NonTzp

    return Prop, Dissp, NonTp
