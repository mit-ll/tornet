"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
CoordConv for tornado detection
"""
import torch
from torch import nn
import torch.nn.modules.conv as conv

class CoordConv2D(nn.Module):
    """
    CoordConv2D layer for working with polar data.

    This module takes a tuple of inputs (image tensor,   image coordinates)
    where,
        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]
    
    This returns a tuple containing the CoordConv convolution and 
    a (possibly downsampled) copy of the coordinate tensor.


    """
    def __init__(self,in_image_channels,
                      in_coord_channels,
                      out_channels,
                      kernel_size,
                      padding='same',
                      stride=1,
                      activation='relu',
                      **kwargs):
        super(CoordConv2D, self).__init__()
        self.n_coord_channels=in_coord_channels
        self.conv = nn.Conv2d(in_image_channels + in_coord_channels, 
                              out_channels,
                              kernel_size, 
                              stride, 
                              padding, **kwargs)
        self.strd=stride
        self.padding=padding
        self.ksize=kernel_size
        if activation is None:
            self.conv_activation=None
        elif activation=='relu':
            self.conv_activation=nn.ReLU()
        else:
            raise NotImplementedError('activation %s not implemented' % activation)
        
    
    def forward(self,inputs):
        """
        inputs is a tuple containing 
          (image tensor,   image coordinates)

        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]

        """
        x,coords=inputs
        x = torch.cat( (x,coords),axis=1)
        x = self.conv(x)
        
        # only apply activation to conv output
        if self.conv_activation:
            x=self.conv_activation(x) 

        # also return coordinates
        if self.padding=='same' and self.strd>1:
            coords = coords[...,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.ksize[0]//2
            if i0>0:
                coords = coords[...,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[...,::self.strd,::self.strd]
        
        return x,coords
        

