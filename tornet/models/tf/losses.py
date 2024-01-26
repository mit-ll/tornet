"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import tensorflow as tf

def _prep(class_labels, logits):
    y_true = tf.cast(class_labels, dtype=logits.dtype)
    y_pred = tf.math.sigmoid(logits) # p=1 means class label=1
    return y_true,y_pred


def mae_loss( class_labels, logits, sample_weights=None ):
    """
    class_labels represents tensor of known binary classes (1,0)
    logits are output of final classification layer that has not yet been run through a sigmoid (from_logits)
    """
    y_true,y_pred=_prep(class_labels, logits)
    if sample_weights is not None:
        denom=tf.reduce_sum(sample_weights)
        return tf.reduce_sum( sample_weights*tf.math.abs(y_true-y_pred) ) / denom
    else:
        return tf.reduce_mean( tf.math.abs(y_true-y_pred) )


def jaccard_loss(class_labels, logits):
    """
    class_labels represents tensor of known binary classes (1,0)
    logits are output of final classification layer that has not yet been run through a sigmoid (from_logits)
    """
    y_true,y_pred=_prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred - intersection
    # Calculate the Jaccard similarity coefficient (IoU)
    iou = intersection / (union + 1e-7)  # Adding a small epsilon to prevent division by zero
    # Jaccard loss is the complement of the Jaccard similarity
    return tf.reduce_mean(1 - iou)


def dice_loss(class_labels, logits):
    """
    y_true:   [Batch, 1]
    y_pred:   [Batch, 1]
    
    """
    y_true,y_pred=_prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return tf.reduce_mean(1.0-dice)
