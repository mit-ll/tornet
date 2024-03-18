# Reformatting TorNet for faster loading

This directory contains tensorflow_dataset `DatasetBuilder` class for tornet.  This reformats data to either `tfrecord` or `array_record`.  These formats provide much faster loading speeds when training deep learning models over the basic loaders that use `xarray`.

To build dataset for TFDS
  
1. Extract all tornet data into a directory `TORNET_ROOT`.  After extraction this directory should contain `catalog.csv, train/, test/`.  This assumes ALL the data was downloaded and extracted.  It that isn't the case, then the `splits` should be adjusted accordingly, but I didn't test running this with only a subset of TorNet.

2. Set env variables      
```
export TORNET_ROOT=/location/of/tornet
export TFDS_DATA_DIR=/where/to/store/tfds_data
```
  
  3. Choose between formats `array_record` (pytorch, JAX) or `tfrecord` (tensorflow).  Run the associated block below.  Note that this process generates a second copy of the dataset (inside `TFDS_DATA_DIR`) and takes a while to run (~24 hours).
  
## For `array_record`

`array_record` is a efficient format that supports random access and generates numpy arrays (so it can be used in torch/JAX/etc.).  This format doesn't require tensorflow at runtime, but tensorflow IS required to build the dataset.  `tensorflow_datasets` and `array_record` are both required at runtime.  For more information, see [https://www.tensorflow.org/datasets/tfless_tfds](https://www.tensorflow.org/datasets/tfless_tfds)

```python
# Builds tornet in array_record format
import os
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder

TORNET_ROOT=os.environ['TORNET_ROOT'] # where tornet files live
TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR'] # where tfds data is to be rewritten

dl_config=tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
tfds.data_source('tornet',
                data_dir=TFDS_DATA_DIR,
                builder_kwargs={'file_format':'array_record'},
                download_and_prepare_kwargs={'download_config':dl_config})
```

The following shows how to use the data at runtime.  This requires `TFDS_DATA_DIR` to be set.

```python
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'

ds = tfds.data_source('tornet')
d = ds['train-2015'][5] # grab a sample
```

## For `tfrecord`

If you're using `tensorflow`, then the dataset can be rewritten to `tfrecord` format.  The resulting dataset can then be used as part of the `td.data` API.

```python
# builds tornet in tfrecord format
import os
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder

TORNET_ROOT=os.environ['TORNET_ROOT'] # where tornet files live
TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR'] # where tfds data is to be rewritten

dl_config=tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
builder = tfds.builder('tornet', 
                       data_dir=TFDS_DATA_DIR, 
                       **{'file_format':'tfrecord'})
builder.download_and_prepare(**{'download_config':dl_config})
```

To use at runtime,

```python
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'

ds = tfds.load('tornet',splits=['train-2013','train-2014'])
```
