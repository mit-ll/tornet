"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
These test confirm the code *runs* and is useful for checking
that the tornet dataset and your python environment are set up properly.
"""


import unittest
import os
import pandas as pd
import xarray as xr

import keras
import logging

if 'TORNET_ROOT' not in os.environ:
    raise RuntimeError('TORNET_ROOT must be defined for unit tests')
DATA_ROOT=os.environ['TORNET_ROOT'] 

try:
    from tornet.data.loader import read_file, TornadoDataLoader
    from tornet.data.keras.loader import KerasDataLoader
    from tornet.data.loader import get_dataloader
    from tornet.models.keras.cnn_baseline import build_model
    from tornet.metrics.keras import metrics as tfm
except ImportError as e:
    print("WARNING: cannot import tornet library. Install it or add to path.")
    raise e



class DataLoading(unittest.TestCase):
    
    @staticmethod
    def read_catalog():
        return pd.read_csv(os.path.join(DATA_ROOT,'catalog.csv'),parse_dates=['start_time','end_time'])

    def test_read_catalog(self):
        DataLoading.read_catalog()
    
    def test_read_file(self):
        catalog = DataLoading.read_catalog().iloc[:100]
        file_list = [os.path.join(DATA_ROOT,f) for f in catalog.filename]
        ds=xr.open_dataset(file_list[0])

        data_loader = TornadoDataLoader(file_list)

        # random access
        data = data_loader[0] 

        # iteration
        for data in data_loader: 
            break
    

class KerasTests(unittest.TestCase):
    def test_keras_dataloader(self):
        ds = KerasDataLoader(data_root=DATA_ROOT,
                            data_type='train',
                            years=[2018,2019],
                            batch_size = 2, 
                            workers = 1,
                            use_multiprocessing = False)
        for x,y in ds:
            break
    
    def _run_train(self,dataloader='keras',**dataloader_kwargs):
        nn = build_model()
        dataloader_kwargs.update({'select_keys':list(nn.input.keys())})
        ds_train = get_dataloader(dataloader, DATA_ROOT, [2014,], "train", 4, **dataloader_kwargs)
        ds_val = get_dataloader(dataloader, DATA_ROOT, [2015,], "train", 4, **dataloader_kwargs)    
        opt  = keras.optimizers.Adam(learning_rate=0.001)
        nn.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=[tfm.BinaryAccuracy(True,name='BinaryAccuracy')])
        history=nn.fit(ds_train, epochs=1, steps_per_epoch=1, 
                       validation_data=ds_val,
                       validation_steps=1,
                       shuffle=False,
                       verbose=0)
        
    def test_training_keras_dataloader(self):
        dataloader_kwargs = {'use_multiprocessing':False,'workers':1}
        self._run_train('keras',**dataloader_kwargs)
        
    def test_training_tf_dataloader(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("tensorflow not installed")
        self._run_train('tensorflow')
    
    def test_training_torch_dataloader(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        self._run_train('torch')
    


if __name__ == '__main__':
    unittest.main()       

    
        