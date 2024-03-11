"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import numpy as np
import tensorflow as tf

from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES

def split_x_y(d):
    """
    Splits dict into X,y, where y are tornado labels
    """
    y=d['ef_number'][None,:]>=0 
    return d,y


def compute_sample_weight(x,y,wN=1.0,w0=1.0,w1=1.0,w2=1.0,wW=0.5):
    """
    Assigns sample weights to samples in x,y based on
    ef_number of tornado
    
    category,  weight
    -----------
    random      wN
    warnings    wW
    0           w0
    1           w1
    2+          w2
    """
    weights = tf.ones_like(y,dtype=tf.float32)
    ef=tf.cast(x['ef_number'],tf.int32)
    warn = x['category']==2 # warnings
    weights = tf.where( ef==-1, wN, weights ) # set all nulls to wN
    weights = tf.where( warn,   wW, weights )   # set warns to wW
    weights = tf.where( ef==0,  w0, weights )
    weights = tf.where( ef==1,  w1, weights )
    weights = tf.where( ef>1,   w2, weights )
    return x,y,weights
