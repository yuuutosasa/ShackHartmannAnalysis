# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# FUNCTION DEFINITION
# -----------------------------------------------------------------------------

def cut2zero(img,cutoff=80):
    ind=img<cutoff
    img[ind]=0
    return img

def moment(grid):
    size=grid.shape[0]
    center=size/2-0.5
    x, y = np.ogrid[0:size, 0:size]

    x=x-center
    y=y-center
    x=x.repeat(size,1)
    y=y.repeat(size,0)
    y*=-1
    
    output=np.zeros((1,2))
    output[...,0]=np.sum(grid*x)/np.sum(grid)/np.max(x)
    output[...,1]=np.sum(grid*y)/np.sum(grid)/np.max(y)

    return output

def shifts(img,grid_size=16,pad_top=8,pad_left=8):
    H,W=img.shape

    N_long=(H-pad_top)//grid_size
    N_lat=(W-pad_left)//grid_size

    shifts=np.full((N_long,N_lat,2), np.nan)

    for i in range(N_long):
        for j in range(N_lat):
            shifts[i,j]=moment(img[pad_top+grid_size*i:pad_top+grid_size-1+grid_size*i,
                                   pad_left+grid_size*j:pad_left+grid_size-1+grid_size*j])
    return shifts
