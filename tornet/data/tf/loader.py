"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Tools for creating tf dataset classes
"""
import os
from typing import List, Dict
import numpy as np
import pandas as pd
import tensorflow as tf

from tornet.data.loader import query_catalog, read_file
from tornet.data.constants import ALL_VARIABLES
from tornet.data import preprocess as pp

def create_tf_dataset(files:str,
                      variables: List[str]=ALL_VARIABLES,
                      n_frames:int=1,
                      tilt_last: bool=True) -> tf.data.Dataset:
    """
    Creates a TF dataset object via the function read_file.   
    This dataset is somewhat slow because of the use of 
    tf.data.dataset.from_generator.  It is recommended to
    use this only as a means to call ds.save() to create a 
    much faster copy of the dataset.
    """
    assert len(files)>0
    # grab one file to gets keys, shapes, etc
    data = read_file(files[0],variables=variables,n_frames=n_frames, tilt_last=tilt_last)
    
    output_signature = { k:tf.TensorSpec(shape=data[k].shape,dtype=data[k].dtype,name=k) for k in data }
    def gen():
        for f in files:
            yield read_file(f,variables=variables,n_frames=n_frames, tilt_last=tilt_last)
    ds = tf.data.Dataset.from_generator(gen,
                                        output_signature=output_signature)
    return ds
    

def shard_function(data: tf.Tensor) -> np.int64:
    """
    Function that "shards" the data in tf.data.Dataset.save().
    This transforms time stamp into a np.int64 between 0,..,9.
    This is optional and may make loading faster by utilizing more CPUs.
    
    """
    x = (data['time'][0]//10) % 10 # uses tens digit of epoch time for shard index
    if x % 10 == 0:
        return np.int64(0)
    elif x % 10 == 1:
        return np.int64(1)
    elif x % 10 == 2:
        return np.int64(2)
    elif x % 10 == 3:
        return np.int64(3)
    elif x % 10 == 4:
        return np.int64(4)
    elif x % 10 == 5:
        return np.int64(5)
    elif x % 10 == 6:
        return np.int64(6)
    elif x % 10 == 7:
        return np.int64(7)
    elif x % 10 == 8:
        return np.int64(8)
    elif x % 10 == 9:
        return np.int64(9)
    else:
        return np.int64(0)



def make_tf_loader(data_root: str, 
            data_type:str='train', # or 'test'
            years: list=list(range(2013,2023)),
            batch_size: int=128, 
            weights: Dict=None,
            include_az: bool=False,
            random_state:int=1234,
            select_keys: list=None,
            tilt_last: bool=True,
            from_tfds: bool=False,
            tfds_data_version: str='1.1.0'):
    """
    Initializes tf.data Dataset for training CNN Tornet baseline.

    data_root - location of TorNet
    data_Type - 'train' or 'test'
    years     - list of years btwn 2013 - 2022 to draw data from
    batch_size - batch size
    weights - optional sample weights, see note below
    include_az - if True, coordinates also contains az field
    random_state - random seed for shuffling files
    select_keys - Only generate a subset of keys from each tornet sample
    tilt_last - If True (default), order of dimensions is left as [batch,azimuth,range,tilt]
                If False, order is permuted to [batch,tilt,azimuth,range]
    from_tfds - Use TFDS data loader, requires this version to be
                built and TFDS_DATA_ROOT to be set.  
                See tornet/data/tdfs/tornet/README.
                If False (default), the basic loader is used
    
    If you leave from_tfds as False, I suggest adding ds=ds.cache( LOCATION ) 
    in the training script to cache the dataset to speed up training times (after epoch 1)
    
    See the DataLoaders.ipynb notebook for details on how to resave TorNet in this way

    weights is optional, if provided must be a dict of the form
      weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    where wN,w0,w1,w2,wW are numeric weights assigned to random,
    ef0, ef1, ef2+ and warnings samples, respectively.  

    After loading TorNet samples, this does the following preprocessing:
    - Optionaly permutes order of dimensions to not have tilt last
    - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
      includes r, r^{-1} (and az if include_az is True)
    - Takes only last time frame
    - Splits sample into inputs,label
    - If weights is provided, returns inputs,label,sample_weights

    """    
    if from_tfds: # fast loader
        import tensorflow_datasets as tfds
        import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
        ds = tfds.load('tornet:%s' % tfds_data_version ,split='+'.join(['%s-%d' % (data_type,y) for y in years]))
        # Assumes data was saved with tilt_last=True and converts it to tilt_last=False
        if not tilt_last:
            ds = ds.map(lambda d: pp.permute_dims(d,(0,3,1,2), backend=tf))
    else: # Load directly from netcdf files
        file_list = query_catalog(data_root, data_type, years, random_state)
        ds = create_tf_dataset(file_list,variables=ALL_VARIABLES,n_frames=1, tilt_last=tilt_last) 

    ds=preproc(ds,weights,include_az,select_keys,tilt_last)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds

def preproc(ds: tf.data.Dataset,
            weights:Dict=None,
            include_az:bool=False,
            select_keys:list=None,
            tilt_last:bool=True):
    """
    Adds preprocessing steps onto dataloader
    """

    # Remove time dimesnion
    ds = ds.map(pp.remove_time_dim)

    # Add coordiante tensors
    ds = ds.map(lambda d: pp.add_coordinates(d,include_az=include_az,tilt_last=tilt_last,backend=tf))

    # split into X,y
    ds = ds.map(pp.split_x_y)

    # Add sample weights
    if weights:
        ds = ds.map(lambda x,y:  pp.compute_sample_weight(x,y,**weights, backend=tf) )
    
        # select keys for input
        if select_keys is not None:
            ds = ds.map(lambda x,y,w: (pp.select_keys(x,keys=select_keys),y,w))
    else:
        if select_keys is not None:
            ds = ds.map(lambda x,y: (pp.select_keys(x,keys=select_keys),y))

    return ds
