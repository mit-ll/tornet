"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""
from typing import Dict
import numpy as np
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


def get_shape(d):
    """
    infers image shape from data in dict d
    """
    # use whatever variable is available
    k = list(set(d.keys()) & set(ALL_VARIABLES)) # assumes this is non-empty!
    return d[k[0]].shape
    
    
def add_coordinates(d,min_range_m=2125.0,
                    include_az=True,
                    tilt_last=True,
                    backend=np):
    """
    Add coordinate tensors r, rinv to data dict d.
    If include_az is True, also add theta.

    Coordinates are stacked along the "tilt" dimension, which is assumed
    to be the final dimension if tilt_last=True.  If tilt_last=False,
    coordinates are concatenated along axis=0.
    
    backend can be np, tf or torch (pass actual imported module)

    min_range_m is minimum possible range of radar data in meters

    """
    full_shape = get_shape(d)
    shape = full_shape[-3:-1] if tilt_last else full_shape[-2:]

    # "250" is the resolution of NEXRAD
    # "1e-5" is scaling applied for normalization
    SCALE = 1e-5 # used to scale range field for CNN
    rng_lower = (d['rng_lower']+250) * SCALE # [1,]
    rng_upper = (d['rng_upper']-250) * SCALE # [1,]
    min_range_m *= SCALE
    
    # Get az range,  convert to math convention where 0 deg is x-axis
    az_lower = d['az_lower']
    az_lower = (90-az_lower) * np.pi/180 # [1,]
    az_upper = d['az_upper']
    az_upper = (90-az_upper) * np.pi/180 # [1,]
    
    # create mesh grids 
    az = backend.linspace( az_lower[0],  az_upper[0], shape[0] )
    rg = backend.linspace( rng_lower[0], rng_upper[0], shape[1] )
    R,A = backend.meshgrid(rg,az,indexing='xy')

    # limit to minimum range of radar
    R = backend.where( R>=min_range_m, R, min_range_m)

    Rinv=1/R
    
    cat_axis = -1 if tilt_last else 0
    if include_az:
        c = backend.stack( (R,A,Rinv), axis=cat_axis )
    else:
        c = backend.stack( (R,Rinv), axis=cat_axis )
    d['coordinates']=c
    return d

def remove_time_dim(d):
    """
    Removes time dimension from data by taking last available frame
    """
    for v in ALL_VARIABLES+['range_folded_mask','label']:
        if v in d:
            d[v]=d[v][-1]
    return d

def add_batch_dim(data: Dict[str,np.ndarray]):
    """
    Adds singleton batch dimension to each array in data
    """
    for k in data:
        data[k] = data[k][None,...]
    return data

def permute_dims(data: Dict[str,np.ndarray], order:tuple):
    """
    Permutes dimensions according to order (see np.transpose)
    Only tensors in data with ndim==len(order) are permuted
    """
    for v in ALL_VARIABLES+['range_folded_mask']:
        if data[v].ndim==len(order):
            data[v]=np.transpose(data[v],order)
    return data
