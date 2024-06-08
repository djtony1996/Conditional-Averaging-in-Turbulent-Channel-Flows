#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:16:47 2024

@author: jitongd
"""

import numpy as np

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




# Example usage
# all_events_array = np.array([0.1, -0.5, 1.6, -0.4, 0.8, -1.5,-1.9,-2, 0.2, -0.7, 2.3])
# k_scale = 1
# detection_case = 1
# whe_single = 1

# pick_events_posi, pick_events_nega = get_detection_events(all_events_array, k_scale, detection_case, whe_single)
# pick_NonTp_posi_index = np.nonzero(pick_events_posi)[0]
# pick_NonTp_nega_index = np.nonzero(pick_events_nega)[0]



