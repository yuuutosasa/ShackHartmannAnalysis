# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# FUNCTION DEFINITION
# -----------------------------------------------------------------------------

def draw_grid(_img, grid=13, grid_size=16, pad_top=8, pad_left=8):
    output=np.copy(_img)
    size=_img.shape[0]
    for i in range(grid+1):
        output[pad_top-1+grid_size*i:pad_top+grid_size*i,:]=[0,0,255]
        output[:,pad_left-1+grid_size*i:pad_left+grid_size*i]=[0,0,255]
        for j in range(grid+1):
            output[pad_top-1+grid_size//2+grid_size*i:pad_top+grid_size//2+grid_size*i, 
                   pad_left-1+grid_size//2+grid_size*j:pad_left+grid_size//2+grid_size*j,...]=[255,0,0]
    return output
    
