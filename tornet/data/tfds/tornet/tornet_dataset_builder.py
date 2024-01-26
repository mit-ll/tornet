"""tornet dataset."""
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

import sys
from tornet.data.loader import read_file

class Builder(tfds.core.GeneratorBasedBuilder):
  """
  DatasetBuilder for tornet.  See README.md in this directory for how to build
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Find instructions to download TorNet on https://github.com/mit-ll/tornet
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'DBZ': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'VEL': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'KDP': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'RHOHV': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'ZDR': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'WIDTH': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'range_folded_mask': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'label': tfds.features.Tensor(shape=(4,),dtype=np.uint8),
            'category': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'event_id': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'ef_number': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'az_lower': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'az_upper': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'rng_lower': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'rng_upper': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'time': tfds.features.Tensor(shape=(4,),dtype=np.int64),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/mit-ll/tornet',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    # Assumes data is already downloaded and extracted from tar files
    # manual_dir should point to where tar files were extracted
    archive_path = dl_manager.manual_dir
    
    # Defines the splits
    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train-2013': self._generate_examples(archive_path / 'train/2013'),
        'train-2014': self._generate_examples(archive_path / 'train/2014'),
        'train-2015': self._generate_examples(archive_path / 'train/2015'),
        'train-2016': self._generate_examples(archive_path / 'train/2016'),
        'train-2017': self._generate_examples(archive_path / 'train/2017'),
        'train-2018': self._generate_examples(archive_path / 'train/2018'),
        'train-2019': self._generate_examples(archive_path / 'train/2019'),
        'train-2020': self._generate_examples(archive_path / 'train/2020'),
        'train-2021': self._generate_examples(archive_path / 'train/2021'),
        'train-2022': self._generate_examples(archive_path / 'train/2022'),
        'test-2013': self._generate_examples(archive_path / 'test/2013'),
        'test-2014': self._generate_examples(archive_path / 'test/2014'),
        'test-2015': self._generate_examples(archive_path / 'test/2015'),
        'test-2016': self._generate_examples(archive_path / 'test/2016'),
        'test-2017': self._generate_examples(archive_path / 'test/2017'),
        'test-2018': self._generate_examples(archive_path / 'test/2018'),
        'test-2019': self._generate_examples(archive_path / 'test/2019'),
        'test-2020': self._generate_examples(archive_path / 'test/2020'),
        'test-2021': self._generate_examples(archive_path / 'test/2021'),
        'test-2022': self._generate_examples(archive_path / 'test/2022')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    # key is the original netcdf filename
    data_type = path.parent.name # 'train' or 'test'
    year = int(os.path.basename(path)) # year
    catalog_path = path / '../../catalog.csv'
    catalog = pd.read_csv(catalog_path,parse_dates=['start_time','end_time'])
    catalog = catalog[catalog['type']==data_type]
    catalog = catalog[catalog.end_time.dt.year.isin([year])]
    catalog = catalog.sample(frac=1,random_state=1234) # shuffle
    #catalog = catalog.iloc[:10] # testing

    for f in catalog.filename:
      # files are relative to dl_manager.manual_dir
      yield f, read_file(path / ('../../'+f),n_frames=4)
