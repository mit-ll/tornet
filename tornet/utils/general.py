"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Utils for importing modules from string
"""

import os
import datetime
import importlib

import tensorflow as tf



def make_exp_dir(exp_dir='../experiments',prefix='',symlink_name='latest',
                 task_type=None, task_id=0):
    """
    Creates a dated directory for an experiement, and also creates a symlink 
    """
    linked_dir=exp_dir+'/%s' % symlink_name
    dated_dir=prefix+'%s-%s-%s' % (datetime.datetime.now().strftime('%y%m%d%H%M%S'),
                                   os.getenv('SLURM_JOB_ID'),
                                   os.getenv('SLURM_ARRAY_TASK_ID'))
    try:
        dated_dir = os.path.join(os.getenv('SLURM_ARRAY_JOB_ID'),dated_dir)
    except:
        pass
    tf.io.gfile.makedirs(os.path.join(exp_dir,dated_dir))
    if os.path.islink(linked_dir):
        os.unlink(linked_dir)
    os.symlink(dated_dir,linked_dir)
    return os.path.join(exp_dir,dated_dir)

def make_callback_dirs(logdir):
    tensorboard_dir = os.path.join(logdir, 'tboard')
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return tensorboard_dir, checkpoints_dir