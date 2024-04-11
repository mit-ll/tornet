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
import tensorflow as tf

class CoordConv2D(tf.keras.layers.Conv2D):
    """
    CoodConv2d layers as described in 

    Liu, Rosanne, et al. "An intriguing failing of convolutional neural networks and 
    the coordconv solution." Advances in neural information processing systems 31 (2018).
    
    """
    def __init__(self,filters,
                      kernel_size,
                      padding='same',
                      strides=(1,1),
                      **kwargs):
        super(CoordConv2D, self).__init__(filters,kernel_size,padding=padding,strides=strides,**kwargs)
        self.ksize=kernel_size
        self.padding=padding
        self.strd=strides[0] # assume equal strides
        self.input_spec = None
    
    def build(self,input_shapes):
        x_shape, coord_shape = input_shapes
        new_shape = x_shape.as_list()
        new_shape[-1] += coord_shape[-1]
        super(CoordConv2D, self).build(new_shape)
        self.input_spec = None

    def call(self, inputs):
        """
        inputs is [N, L, W, C] tensor
        coords is [N, L, W, nd] tensor of coordiantes
        """
        x, coords = inputs
        # Stack x with coordinates
        x = tf.concat( (x,coords), axis=3)
        
        # Run convolution
        conv=super(CoordConv2D, self).call(x)
        
        # The returned coordinates should have same shape as conv 
        # prep the coordiantes by slicing them to the same shape  
        # as conv
        if self.padding=='same' and self.strd>1:
            coords = coords[:,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.ksize[0]//2
            if i0>0:
                coords = coords[:,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[:,::self.strd,::self.strd]
        
        return conv,coords

