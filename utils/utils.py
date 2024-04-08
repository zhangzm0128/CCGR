import torch 
import torch.nn as nn
import numpy as np
    
COLOR_LIMIT = [-50, 0]
EPSLION = 1e-10
    
def format_runtime(time_gap):
    m, s = divmod(time_gap, 60)
    h, m = divmod(m, 60)
    runtime_str = ''
    if h != 0:
        runtime_str = runtime_str + '{}h'.format(int(h))
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    elif m != 0:
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    else:
        runtime_str = runtime_str + '{:.4}s'.format(s)
    return runtime_str

def generate_bmode(image_data):
    bmode = 20 * np.log10(image_data / np.max(image_data))
    bmode = np.clip(bmode, COLOR_LIMIT[0], COLOR_LIMIT[1])
    return bmode
    
def norm_data(data, range_values):
    nr_max = range_values[1]
    nr_min = range_values[0]
    x_scaled = (data - np.min(data)) / (np.max(data) - np.min(data) + EPSLION) * (nr_max - nr_min) + nr_min
        
    return x_scaled

    
