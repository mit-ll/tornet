# TorNet

Software to work with the TorNet dataset as described in the paper [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437)

![Alt text](tornet_image.png?raw=true "sample")

## Downloading the Data

The TorNet dataset can be downloaded from the following location:

#### Zenodo

TorNet is split across 10 files, each containing 1 year of data. There is also a catalog CSV file that is used by some functions in this repository.    

* Tornet 2013 (3 GB) and catalog: [https://zenodo.org/doi/10.5281/zenodo.10558658](https://zenodo.org/doi/10.5281/zenodo.10558658)
* Tornet 2014 (15 GB): [https://zenodo.org/doi/10.5281/zenodo.10558838](https://zenodo.org/doi/10.5281/zenodo.10558838)
* Tornet 2015 (17 GB): [https://zenodo.org/doi/10.5281/zenodo.10558853](https://zenodo.org/doi/10.5281/zenodo.10558853)
* Tornet 2016 (16 GB): [https://zenodo.org/doi/10.5281/zenodo.10565458](https://zenodo.org/doi/10.5281/zenodo.10565458)
* Tornet 2017 (15 GB): [https://zenodo.org/doi/10.5281/zenodo.10565489](https://zenodo.org/doi/10.5281/zenodo.10565489)
* Tornet 2018 (12 GB): [https://zenodo.org/doi/10.5281/zenodo.10565514](https://zenodo.org/doi/10.5281/zenodo.10565514)
* Tornet 2019 (18 GB): [https://zenodo.org/doi/10.5281/zenodo.10565535](https://zenodo.org/doi/10.5281/zenodo.10565535)
* Tornet 2020 (17 GB): [https://zenodo.org/doi/10.5281/zenodo.10565581](https://zenodo.org/doi/10.5281/zenodo.10565581)
* Tornet 2021 (18 GB): [https://zenodo.org/doi/10.5281/zenodo.10565670](https://zenodo.org/doi/10.5281/zenodo.10565670)
* Tornet 2022 (19 GB): [https://zenodo.org/doi/10.5281/zenodo.10565691](https://zenodo.org/doi/10.5281/zenodo.10565691)

If downloading through your browser is slow, we recommend downloading these using `zenodo_get` (https://gitlab.com/dvolgyes/zenodo_get).

After downloading, there should be 11 files, `catalog.csv`, and 10 files named as `tornet_YYYY.tar.gz`.   Move and untar these into a target directory, which will be referenced using the `TORNET_ROOT` environment variable in the code.  After untarring the 10 files, this directory should contain `catalog.csv` along with sub-directories `train/` and `test/` filled with `.nc` files for each year in the dataset.


## Setup

Basic python requirements are listed in `requirements/basic.txt` and can be installed using `pip install -r requirements.txt`.

The `tornet` package can then installed into your environment by running

`pip install .`

in this repo.  To do ML with TorNet, additional installs may be necessary depending on library of choice.  See e.g., `requirements/tensorflow.txt`, `requirements/torch.txt`.

## Loading and visualizing TorNet

Start with `notebooks/DataLoaders.ipynb` to get an overview on loading and visualizing the dataset.

## Train CNN baseline model

The following trains the CNN baseline model described in the paper using `tensorflow`.  If you run this out-of-the-box, it will run very slowly because it uses the basic dataloader.  Read the DataLoader notebook for tips on how to optimize the data loader.
```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet     

# Run training
python scripts/tornado_detection/train_tornado_tf.py scripts/tornado_detection/config/params.json
```

## Evaluate trained model
Weights of a pretrained CNN baseline are provided in `model/`.  To evaluate this model on the test set, run

```
# Set path to dataset
export TORNET_ROOT=/path/to/tornet  

# Evaluate trained model
python scripts/tornado_detection/test_tornado_tf.py models/tornado_detector_baseline.SavedModel
```

This will compute and print various metrics computed on the test set.


### Disclosure
```
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
```
