"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import math
import os
from typing import Dict

import keras
import numpy as np
import pandas as pd

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import query_catalog, read_file


class KerasDataLoader(keras.utils.PyDataset):
    """Backend-agnostic dataloader that produces dictionaries of numpy arrays

    Supports multiprocessing because of inheritance from keras.utils.PyDataset.
    Multiprocessing is only used within keras.Model.fit() and not when iterating
    through the dataloader outside of keras.Model.fit()

    After loading TorNet samples, this does the following preprocessing:
    - Optionaly permutes order of dimensions to not have tilt last
    - Takes only last time frame
    - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
      includes r, r^{-1} (and az if include_az is True)
    - Splits sample into inputs,label
    - If weights is provided, returns inputs,label,sample_weights
    """

    def __init__(
        self,
        data_root: str,
        data_type: str = "train",
        years: list = list(range(2013, 2023)),
        catalog: pd.DataFrame=None,
        batch_size: int = 128,
        weights: Dict = None,
        include_az: bool = False,
        random_state: int = 1234,
        select_keys: list = None,
        tilt_last: bool = True,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
    ):
        """
        data_root - location of TorNet
        data_Type - 'train' or 'test'
        years     - list of years btwn 2013 - 2022 to draw data from
        catalog   - preloaded catalog (optional)
        batch_size - batch size
        weights - optional sample weights, see note below
        include_az - if True, coordinates also contains az field
        random_state - random seed for shuffling files
        select_keys - only generate a subset of keys from each tornet sample
        tilt_last - if True (default), order of dimensions is left as 
            [batch,azimuth,range,tilt] If False, order is permuted to 
            [batch,tilt,azimuth,range]
        workers, use_multiprocessing, max_queue_size - see:
        https://keras.io/api/utils/python_utils/#pydataset-class

        When workers==0, workers becomes os.cpu_count().

        """

        if workers == 0:
            workers = os.cpu_count()

        super().__init__(workers, use_multiprocessing, max_queue_size)
        self.data_root = data_root
        self.data_type = data_type
        self.years = years
        self.batch_size = batch_size
        self.weights = weights
        self.include_az = include_az
        self.random_state = random_state
        self.select_keys=select_keys

        self.tilt_last = tilt_last
        self.file_list = query_catalog(data_root, data_type, years, random_state, catalog=catalog)

    def __len__(self) -> int:
        "Returns number of batches"
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        "Returns a batch of data"
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.file_list))

        files_batch = self.file_list[low:high]

        element_list = []
        for f in files_batch:
            element_list.append(read_file(f, variables=ALL_VARIABLES, n_frames=1, tilt_last=self.tilt_last))

        # Transforms
        for element in element_list:
            pp.add_coordinates(element, include_az=self.include_az, backend=np, tilt_last=self.tilt_last)

        # Add batch dimension to coordinates
        for el in element_list:
            el["coordinates"] = el["coordinates"][None, ...]

        # Concatenate into batch
        batch = {}
        for key in element_list[0].keys():
            batch[key] = np.concatenate([el[key] for el in element_list])
        
        # split into x,y
        x, y = pp.split_x_y(batch)

        if self.weights:
            x, y, w = pp.compute_sample_weight(x, y, **self.weights, backend=np)
            return pp.select_keys(x,keys=self.select_keys),y,w
        else:
            return pp.select_keys(x,keys=self.select_keys), y
