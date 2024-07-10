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
import json
import shutil
import keras

import logging
logging.basicConfig(level=logging.INFO)

from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES

from tornet.models.keras.losses import mae_loss

from tornet.models.keras.cnn_baseline import build_model

from tornet.metrics.keras import metrics as tfm

from tornet.utils.general import make_exp_dir, make_callback_dirs

EXP_DIR=os.environ.get('EXP_DIR','.')
DATA_ROOT=os.environ['TORNET_ROOT']
logging.info('TORNET_ROOT='+DATA_ROOT)

DEFAULT_CONFIG={
    'epochs':10,
    'input_variables':ALL_VARIABLES,
    'train_years':list(range(2013,2021)),
    'val_years':list(range(2021,2023)),
    'batch_size':128,
    'model':'vgg',
    'start_filters':48,
    'learning_rate':1e-4,
    'decay_steps':1386,
    'decay_rate':0.958,
    'l2_reg':1e-5,
    'wN':1.0,
    'w0':1.0,
    'w1':1.0,
    'w2':2.0,
    'wW':0.5,
    'label_smooth':0,
    'loss':'cce',
    'head':'maxpool',
    'exp_name':'tornet_baseline',
    'exp_dir':EXP_DIR,
    'dataloader':"keras",
    'dataloader_kwargs': {}
}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate=config.get('learning_rate')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
    label_smooth=config.get('label_smooth')
    loss_fn = config.get('loss')
    input_variables=config.get('input_variables')
    exp_name=config.get('exp_name')
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f'Using {dataloader} dataloader')
    logging.info('Running with config:')
    logging.info(config)

    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    
    # Create data laoders
    dataloader_kwargs.update({'select_keys':input_variables+['range_folded_mask','coordinates']})
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs)    
    
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
    
    nn = build_model(shape=in_shapes,
                     c_shape=c_shapes,
                     start_filters=start_filters,
                     l2_reg=l2_reg,
                     input_variables=input_variables,
                     head=head)
    
    # model setup
    lr=keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay")
    
    from_logits=True
    if loss_fn.lower()=='cce':
        loss = keras.losses.BinaryCrossentropy( from_logits=from_logits, 
                                                    label_smoothing=label_smooth )
    elif loss_fn.lower()=='hinge':
        loss = keras.losses.Hinge() # automatically converts labels to -1,1
    elif loss_fn.lower()=='mae':
        loss = lambda yt,yp: mae_loss(yt,yp)
    else:
        raise RuntimeError('unknown loss %s' % loss_fn)


    opt  = keras.optimizers.Adam(learning_rate=lr)

    # Compute various metrics while training
    metrics = [keras.metrics.AUC(from_logits=from_logits,name='AUC',num_thresholds=2000),
                keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    
    nn.compile(loss=loss,
                metrics=metrics,
                optimizer=opt,
                weighted_metrics=[])
    
    ## Setup experiment directory and model callbacks
    expdir = make_exp_dir(exp_dir=exp_dir,prefix=exp_name)
    logging.info('expdir='+expdir)

    # Copy the properties that were used
    with open(os.path.join(expdir,'data.json'),'w') as f:
        json.dump(
            {'data_root':DATA_ROOT,
             'train_data':list(train_years), 
             'val_data':list(val_years)},f)
    with open(os.path.join(expdir,'params.json'),'w') as f:
        json.dump({'config':config},f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir,'train.py')) 
    
    # Callbacks
    tboard_dir, checkpoints_dir=make_callback_dirs(expdir)
    checkpoint_name=os.path.join(checkpoints_dir, 'tornadoDetector'+'_{epoch:03d}.keras' )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name,monitor='val_loss',save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir,'history.csv')),
        keras.callbacks.TerminateOnNaN(),
    ]

    # TensorBoard callback requires tensorflow backend
    if keras.config.backend() == "tensorflow":
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tboard_dir,write_graph=False))#,profile_batch=(5,15)),

    ## FIT
    history=nn.fit(ds_train,
                   epochs=epochs,
                    validation_data=ds_val,
                    callbacks=callbacks,
                    verbose=1) 
    
    # At the end,  report the best score observed over all epochs
    if len(history.history['val_AUC'])>0:
        best_auc = np.max(history.history['val_AUC'])
        best_aucpr = np.max(history.history['val_AUCPR'])
    else:
        best_auc,best_aucpr=0.5,0.0
    
    return {'AUC':best_auc,'AUCPR':best_aucpr}


if __name__=='__main__':
    config=DEFAULT_CONFIG
    # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
    main(config)
