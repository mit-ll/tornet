"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import matplotlib
import numpy as np
import tensorflow as tf

from tornet.data.tf.loader import create_tf_dataset
from tornet.data.constants import ALL_VARIABLES
from tornet.data.tf import preprocess as pp 
from tornet.display.tboard import log_image

class LogTornadoImage(tf.keras.callbacks.Callback):
    """
    Creates GSWR images for tensorboard
    """
    def __init__(self, 
                 filenames,
                 tboard_dir,
                 vars_to_plot=ALL_VARIABLES,
                 include_az=True,
                 **kwargs):
        """
        Creates images in tensorboard with assigned classification scores
        """
        super(LogTornadoImage,self).__init__(**kwargs)
        self.filenames=filenames

        # Make dataloader
        ds =create_tf_dataset(self.filenames,
                               variables=ALL_VARIABLES,
                               n_frames=1)
        ds = ds.map(lambda d: pp.add_coordinates(d,include_az=include_az))
        ds = ds.map(pp.remove_time_dim)
        ds = ds.map(pp.split_x_y)
        ds = ds.batch(1)
        self.ds=ds

        self.tboard_dir=tboard_dir
        self.vars_to_plot=vars_to_plot
        self.file_writer = tf.summary.create_file_writer(tboard_dir+"/test_images/")
        matplotlib.use('Agg')
    
    def on_epoch_end( self, epoch, logs=None):
        
        # for each file, run prediction
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        for (x,y),fname in zip(self.ds,self.filenames):
            score = sigmoid(self.model.predict(x,verbose=0))
            log_image(x, score, fname, ALL_VARIABLES, self.file_writer, epoch)


