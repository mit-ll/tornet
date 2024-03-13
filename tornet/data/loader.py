"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Tools to read tornado samples
"""
from typing import Dict, List, Callable
import numpy as np
import xarray as xr

from tornet.data.constants import ALL_VARIABLES

def read_file(f: str, 
              variables: List['str']=ALL_VARIABLES,
              n_frames:int=1,
              tilt_last:bool=True) -> Dict[str,np.ndarray]:
    """
    Extracts data from a single netcdf file

    Inputs:
    f: nc filename
    variables:  List of radar variables to load.  
                Default is all 6 variables ['DBZ','VEL','KDP','RHOHV','ZDR','WIDTH']
    n_frames:  number of frames to use. 1=last frame only.  No more than 4
               Default is 1.
    tilt_last:  If True (default), order of dimensions is left as [time,azimuth,range,tilt]
                If False, order is permuted to [time,tilt,azimuth,range]

    Returns:
    Dict containing data for each variable, along with several metadata fields.
    """
    
    data = {}
    with xr.open_dataset(f) as ds:
        
        # Load each radar variable
        for v in variables:
            data[v]=ds[v].values[-n_frames:,:,:,:]
        
        # Various numeric metadata
        data['range_folded_mask'] = ds['range_folded_mask'].values[-n_frames:,:,:,:].astype(np.float32) # only two channels for vel,width
        data['label'] = ds['frame_labels'].values[-n_frames:] # 1 if tornado, 0 otherwise
        data['category']=np.array([{'TOR':0,'NUL':1,'WRN':2}[ds.attrs['category']]]) # tornadic, null (random), or warning
        data['event_id']=np.array([int(ds.attrs['event_id'])])
        data['ef_number']=np.array([int(ds.attrs['ef_number'])])
        data['az_lower']=np.array(ds['azimuth_limits'].values[0:1])
        data['az_upper']=np.array(ds['azimuth_limits'].values[1:])
        data['rng_lower']=np.array(ds['range_limits'].values[0:1])
        data['rng_upper']=np.array(ds['range_limits'].values[1:])
        data['time']=(ds.time.values[-n_frames:].astype(np.int64)/1e9).astype(np.int64)

    # Fix for v1 of the data
    # Make sure final label is consistent with ef_number 
    data['label'][-1] = (data['ef_number'][0]>=0)
    
    if not tilt_last: 
        for v in variables+['range_folded_mask']:
            data[v]=np.transpose(data[v],(0,3,1,2))
        
    return data



class TornadoDataLoader:
    """
    Tornado data loader class
    
    file_list:    list of TorNet filenames to load
    variables: list of TorNet variables to load (subset of ALL_VARIABLES)
    n_frames:  number of time frames to load (ending in last frame)
    shuffle:   If True, shuffles file_list before loading
    tilt_last:  If True (default), order of dimensions is left as [time,azimuth,range,tilt]
                If False, order is permuted to [time,tilt,azimuth,range]
                (if other dim orders are needed, use a transform)
    transform:  If provided, this callable is applied to transform each sample 
                before being returned

    """
    def __init__(self,
                 file_list:List[str],
                 variables: List['str']=ALL_VARIABLES,
                 n_frames:int=1,
                 shuffle:bool=False,
                 tilt_last:bool=True,
                 transform:Callable=None 
                 ):
        if shuffle:
            np.random.shuffle(file_list)
        self.file_list=file_list
        self.variables=variables
        self.n_frames=n_frames
        self.tilt_last=tilt_last
        self.current_file_index=0
        self.transform=transform
    def __iter__(self):
        self.current_file_index=0
        return self
    def __next__(self):
        if self.current_file_index<len(self):
            self.current_file_index+=1
            return self[self.current_file_index]
        else:
            raise StopIteration
    def __getitem__(self,index:int):
        """
        Reads file at index
        """
        data = read_file(self.file_list[index],
                         variables=self.variables,
                         tilt_last=self.tilt_last,
                         n_frames=self.n_frames)

        if self.transform:
            data = self.transform(data)
        return data
       
    def __len__(self):
        return len(self.file_list)
    



