"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Custom layers for tornado detection
"""
import keras
from keras import ops

@keras.saving.register_keras_serializable()
class CoordConv2D(keras.layers.Layer):
    """
    Adopted from the CoodConv2d layers as described in 

    Liu, Rosanne, et al. "An intriguing failing of convolutional neural networks and 
    the coordconv solution." Advances in neural information processing systems 31 (2018).
    
    """
    def __init__(self,filters,
                      kernel_size,
                      kernel_regularizer,
                      activation,
                      padding='same',
                      strides=(1,1),
                      conv2d_kwargs = {},
                      **kwargs):

        super(CoordConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.conv2d_kwargs = conv2d_kwargs
        self.strd = strides[0]  # assume equal strides

        self.conv = keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            kernel_regularizer=self.kernel_regularizer,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides,
            **conv2d_kwargs
        )

    def build(self, input_shape):
        x_shape, coord_shape = input_shape
        concat_shape = list(x_shape)
        concat_shape[-1] += coord_shape[-1]
        self.conv.build(concat_shape)

    def call(self,inputs):
        """
        inputs is a tuple 
           [N, L, W, C] data tensor,
           [N, L, W, nd] tensor of coordiantes
        """
        x, coords = inputs

        # Stack x with coordinates
        x = ops.concatenate( (x,coords), axis=-1)

        # Run convolution
        conv=self.conv(x)

        # The returned coordinates should have same shape as conv
        # prep the coordiantes by slicing them to the same shape
        # as conv
        if self.padding=='same' and self.strd>1:
            coords = coords[:,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.kernel_size[0]//2
            if i0>0:
                coords = coords[:,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[:,::self.strd,::self.strd]

        return conv,coords

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update(
            {   "filters": self.filters,
                "kernel_size": self.kernel_size,
                "kernel_regularizer": self.kernel_regularizer,
                "activation":self.activation,
                "padding": self.padding,
                "strides": self.strides,
                "conv2d_kwargs": self.conv2d_kwargs
            }
        )
        return config

@keras.saving.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    """Fill NaNs with fill_val"""
    def __init__(self, fill_val, **kwargs):
        super(FillNaNs, self).__init__(**kwargs)
        self.fill_val = fill_val

    def __call__(self, x):
        return ops.where(ops.isnan(x), self.fill_val, x)

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update({"fill_val": self.fill_val})
        return config
