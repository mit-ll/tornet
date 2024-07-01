"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import sys

import os
import numpy as np
import pandas as pd
import keras
import tqdm

from tornet.data.tf.loader import make_ds
from tornet.metrics.keras import metrics as tfm

import logging
logging.basicConfig(level=logging.INFO)

data_root=os.environ['TORNET_ROOT']
logging.info('TORNET_ROOT='+data_root)

# Assume we are using tfds if TFDS_DATA_DIR is defined
from_tfds=('TFDS_DATA_DIR' in os.environ)
if from_tfds:
    logging.info('Using TFDS dataset location at '+os.environ['TFDS_DATA_DIR'])

FILTER_WARNINGS=False

def main():

    trained_model = sys.argv[1]

    ## Set up data loaders
    test_years = range(2013,2023)
    ds_test = make_ds(data_root,
                      data_type='test',
                      years=test_years,
                      batch_size=64,
                      filter_warnings=FILTER_WARNINGS,
                      include_az=False,
                      from_tfds=from_tfds)  
    
    model = keras.saving.load_model(trained_model,compile=False)
    
    # limit dataset to only required model inputs
    ds_test = ds_test.map(\
        lambda x,y: ({k:x[k] for k in [i.name for i in model.inputs]},y)
        )

    # Compute various metrics
    from_logits=True
    metrics = [ keras.metrics.AUC(from_logits=from_logits,name='AUC',num_thresholds=2000),
                keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                tfm.BinaryAccuracy(from_logits=from_logits,name='BinaryAccuracy'), 
                tfm.Precision(from_logits=from_logits,name='Precision'), 
                tfm.Recall(from_logits=from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    model.compile(metrics=metrics)

    scores = model.evaluate(ds_test) 
    scores = {m.name:scores[k+1] for k,m in enumerate(metrics)}

    logging.info(scores)

 
if __name__=='__main__':
    main()
