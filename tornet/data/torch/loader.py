"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import tornet.data.preprocess as pp
from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import TornadoDataLoader, query_catalog


def numpy_to_torch(d: Dict[str, np.ndarray]):
    for key, val in d.items():
        d[key] = torch.from_numpy(np.array(val))
    return d

def make_torch_loader(data_root: str, 
                data_type:str='train', # or 'test'
                years: list=list(range(2013,2023)),
                batch_size: int=128, 
                weights: Dict=None,
                include_az: bool=False,
                random_state:int=1234,
                select_keys: list=None,
                tilt_last: bool=True,
                from_tfds: bool=False,
                workers:int=8):
    """
    Initializes torch.utils.data.DataLoader for training CNN Tornet baseline.

    data_root - location of TorNet
    data_Type - 'train' or 'test'
    years     - list of years btwn 2013 - 2022 to draw data from
    batch_size - batch size
    weights - optional sample weights, see note below
    include_az - if True, coordinates also contains az field
    random_state - random seed for shuffling files
    workers - number of workers to use for loading batches
    select_keys - Only generate a subset of keys from each tornet sample
    tilt_last - If True (default), order of dimensions is left as [batch,azimuth,range,tilt]
                If False, order is permuted to [batch,tilt,azimuth,range]
    from_tfds - Use TFDS data loader, requires this version to be
                built and TFDS_DATA_ROOT to be set.  
                See tornet/data/tdfs/tornet/README.
                If False (default), the basic loader is used

    weights is optional, if provided must be a dict of the form
      weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    where wN,w0,w1,w2,wW are numeric weights assigned to random,
    ef0, ef1, ef2+ and warnings samples, respectively.  

    After loading TorNet samples, this does the following preprocessing:
    - Optionaly permutes order of dimensions to not have tilt last
    - Takes only last time frame
    - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
      includes r, r^{-1} (and az if include_az is True)
    - Splits sample into inputs,label
    - If weights is provided, returns inputs,label,sample_weights
    """    
    if from_tfds:
        import tensorflow_datasets as tfds
        import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
        ds = tfds.data_source('tornet')

        transform_list = []

        # Assumes data was saved with tilt_last=True and converts it to tilt_last=False
        if not tilt_last:
            transform_list.append(lambda d: pp.permute_dims(d,(0,3,1,2)))

        transform_list.append(
            lambda d: pp.remove_time_dim(d),
            lambda d: pp.add_coordinates(d, include_az=include_az, tilt_last=tilt_last, backend=torch),
            lambda d: pp.split_x_y(d)
        )

        if weights:
            transform_list.append(lambda xy: pp.compute_sample_weight(*xy, **weights, backend=torch))
        
        if select_keys is not None:
            transform_list.append(
                lambda xy: pp.select_keys(xy[0],keys=select_keys)+xy[1:]
            )
            
         # Dataset, with preprocessing
        transform = transforms.Compose(transform_list)

        datasets = [TFDSTornadoDataset(ds['%s-%d' % (data_type,y)] ,transform) for y in years]
        dataset = torch.utils.data.ConcatDataset(datasets)

    else:
        file_list = query_catalog(data_root, data_type, years, random_state)

        transform_list = [
            lambda d: numpy_to_torch(d),
            lambda d: pp.remove_time_dim(d),
            lambda d: pp.add_coordinates(d, include_az=include_az, tilt_last=tilt_last, backend=torch),
            lambda d: pp.split_x_y(d),
        ]

        if weights:
            transform_list.append(lambda xy: pp.compute_sample_weight(*xy, **weights, backend=torch))
        
        if select_keys is not None:
            transform_list.append(
                lambda xy: (pp.select_keys(xy[0],keys=select_keys),)+xy[1:]
            )

        # Dataset, with preprocessing
        transform = transforms.Compose(transform_list)

        dataset = TornadoDataset(file_list,
                                 variables=ALL_VARIABLES,
                                 n_frames=1,
                                 tilt_last=tilt_last,
                                 transform=transform)

    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=workers)
    return loader

    
class TornadoDataset(TornadoDataLoader,Dataset):
    pass

class TFDSTornadoDataset(Dataset):
    def __init__(self,ds,transforms=None):
          self.ds=ds
          self.transforms=transforms
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds.__getitem__(idx)
        if self.transforms:
             x=self.transforms(x)
        return x
