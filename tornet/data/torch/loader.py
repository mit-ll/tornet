"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import pandas as pd

import torch
from torch import optim, nn
from torch.utils.data import Dataset
from torchvision import transforms, utils

from tornet.data.loader import TornadoDataLoader
from tornet.data.preprocess import add_coordinates, remove_time_dim, permute_dims
from tornet.data.constants import ALL_VARIABLES

def make_loader(data_root: str, 
                data_type:str='train', # or 'test'
                years: list=list(range(2013,2023)),
                batch_size: int=128, 
                include_az: bool=False,
                random_state:int=1234,
                num_workers:int=8,
                from_tfds: bool=False):
    """
    Initializes tf.data Dataset for training CNN Tornet baseline.

    data_root - location of TorNet
    data_Type - 'train' or 'test'
    years     - list of years btwn 2013 - 2022 to draw data from
    batch_size - batch size
    include_az - if True, coordinates also contains az field
    random_state - random seed for shuffling files
    num_workers - number of workers to use for loading batches
    from_tfds - Use TFDS data loader, requires this version to be
                built and TFDS_DATA_ROOT to be set.  
                See tornet/data/tdfs/tornet/README.
                If False (default), the basic loader is used

    After loading TorNet samples, this does the following preprocessing:
    - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
      includes r, r^{-1} (and az if include_az is True)
    - Takes only last time frame

    """    
    if from_tfds:
        import tensorflow_datasets as tfds
        import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
        ds = tfds.data_source('tornet')
        # Dataset, with preprocessing
        transform = transforms.Compose([
                    # transpose to [time,tile,az,rng]
                    lambda d: permute_dims(d,(0,3,1,2)),
                    # add coordinates tensor to data
                    lambda d: add_coordinates(d,include_az=include_az,tilt_last=False,backend=torch), 
                    # Remove time dimension
                    lambda d: remove_time_dim(d)])                                
        datasets = [
            TFDSTornadoDataset(ds['%s-%d' % (data_type,y)] ,transform)  for y in years
            ]
        dataset = torch.utils.data.ConcatDataset(datasets)

    else:
        catalog_path = os.path.join(data_root,'catalog.csv')
        if not os.path.exists(catalog_path):
            raise RuntimeError('Unable to find catalog.csv at '+data_root)
                
        catalog = pd.read_csv(catalog_path,parse_dates=['start_time','end_time'])
        catalog = catalog[catalog['type']==data_type]
        catalog = catalog[catalog.start_time.dt.year.isin(years)]
        catalog = catalog.sample(frac=1,random_state=random_state) # shuffles list
        file_list = [os.path.join(data_root,f) for f in catalog.filename]

        # Dataset, with preprocessing
        transform = transforms.Compose([
                    # add coordinates tensor to data
                    lambda d: add_coordinates(d,include_az=False,tilt_last=False,backend=torch), 
                    # Remove time dimension
                    lambda d: remove_time_dim(d)])                                
        dataset = TornadoDataset(file_list,
                                variables=ALL_VARIABLES,
                                n_frames=1,
                                tilt_last=False, # so ordering of dims is [time,tilt,az,range]
                                transform=transform) 
    loader = torch.utils.data.DataLoader( dataset, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers )
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
     
      