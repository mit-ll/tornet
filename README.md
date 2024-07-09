# TorNet

Software to work with the TorNet dataset as described in the paper [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437)

## Updates (7/9/24)

* The TorNet dataset has been updated to version 1.1.  This update fixes a small number of incorrectly labeled frames in v1 of the dataset, and fixes the event and episode IDs of the warning categoies.   Version 1.1 also provides the tornado start and end times in the metadata.  We recommend re-downloading the newer version of the data using the links below. 

* The code and pretrained models are now compatible with `keras` 3.0.   Users can now select their deep learning backend from `tensorflow`, `torch`, or `jax`.  Backend-agnostic data loaders are also provided.  Read more about this library at [keras's website](https://keras.io/).  Users of `tf.keras` should use the `tf_keras` branch of this repo.  

* The pretrained CNN model is now available on [huggingface (tornet-ml/tornado_detector_baseline_v1)](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1).   Instructions for downloading and using the pre-trained model can be found in `models/README.md` and in the `VisualizeSamples.ipynb` notebook.


![Alt text](tornet_image.png?raw=true "sample")



## Downloading the Data

The TorNet dataset can be downloaded from the following location:

#### Zenodo

TorNet is split across 10 files, each containing 1 year of data. There is also a catalog CSV file that is used by some functions in this repository.    

* Tornet 2013 (3 GB) and catalog: [https://doi.org/10.5281/zenodo.12636522](https://doi.org/10.5281/zenodo.12636522)
* Tornet 2014 (15 GB): [https://doi.org/10.5281/zenodo.12637032](https://doi.org/10.5281/zenodo.12637032)
* Tornet 2015 (17 GB): [https://doi.org/10.5281/zenodo.12655151](https://doi.org/10.5281/zenodo.12655151)
* Tornet 2016 (16 GB): [https://doi.org/10.5281/zenodo.12655179](https://doi.org/10.5281/zenodo.12655179)
* Tornet 2017 (15 GB): [https://doi.org/10.5281/zenodo.12655183](https://doi.org/10.5281/zenodo.12655183)
* Tornet 2018 (12 GB): [https://doi.org/10.5281/zenodo.12655187](https://doi.org/10.5281/zenodo.12655187)
* Tornet 2019 (18 GB): [https://doi.org/10.5281/zenodo.12655716](https://doi.org/10.5281/zenodo.12655716)
* Tornet 2020 (17 GB): [https://doi.org/10.5281/zenodo.12655717](https://doi.org/10.5281/zenodo.12655717)
* Tornet 2021 (18 GB): [https://doi.org/10.5281/zenodo.12655718](https://doi.org/10.5281/zenodo.12655718)
* Tornet 2022 (19 GB): [https://doi.org/10.5281/zenodo.12655719](https://doi.org/10.5281/zenodo.12655719)

If downloading through your browser is slow, we recommend downloading these using `zenodo_get` (https://gitlab.com/dvolgyes/zenodo_get).

After downloading, there should be 11 files, `catalog.csv`, and 10 files named as `tornet_YYYY.tar.gz`.   Move and untar these into a target directory, which will be referenced using the `TORNET_ROOT` environment variable in the code.  After untarring the 10 files, this directory should contain `catalog.csv` along with sub-directories `train/` and `test/` filled with `.nc` files for each year in the dataset.


## Setup

Basic python requirements are listed in `requirements/basic.txt`.

The `tornet` package can then installed into your environment by running

`pip install .`

In this repo.  To do ML with TorNet, additional installs may be necessary depending on library of choice.  See e.g., `requirements/tensorflow.txt`, `requirements/torch.txt` and/or `requirements/jax.txt`.  

Please note that we did not exhaustively test all combinations of operating systems, data loaders, deep learning frameworks, and GPU usage.  If you are using the latest version of `keras`, then I recommend you follow setup instructions on the keras webpage [https://keras.io/getting_started/](https://keras.io/getting_started/).  Feel free to describe any issues you are having under the issues tab.  

### Conda

If using conda

```
conda create -n tornet-{backend} python=3.10
conda activate tornet-{backend}
pip install -r requirements/{backend}.txt
```

Replace {backend} with tensorflow, torch or jax.


## Loading and visualizing TorNet

Start with `notebooks/DataLoaders.ipynb` to get an overview on loading and visualizing the dataset.

To run inference on TorNet samples using a pretrained model,  look at `notebooks/VisualizeSamples.ipynb`.

## Train CNN baseline model

### Multiple backend support with Keras 3
The model uses Keras 3 which supports multiple backends. The environment variable
KERAS_BACKEND can be used to choose the backend. 

```
export KERAS_BACKEND=tensorflow
# export KERAS_BACKEND=torch
# export KERAS_BACKEND=jax
```

The following trains the CNN baseline model described in the paper using `tensorflow`.  If you run this out-of-the-box, it will run very slowly because it uses the basic dataloader.  Read the DataLoader notebook for tips on how to optimize the data loader.
```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet     

# Run training
python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params.json
```

## Evaluate trained model
To evaluate this model on the test set, run

```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet  

# Evaluate trained model
python scripts/tornado_detection/test_tornado_keras.py 
```

This will compute and print various metrics computed on the test set.  Note that this script will attempt to download pretrained weights from huggingface, so ensure there is internet connectivity.  Alternatively, manually download the pretrained yourself and provide with `--model_path`


### Disclosure
```
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
```
