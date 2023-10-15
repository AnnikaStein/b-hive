# FROM https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        #assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    # (||^2 + ... ||^2) ^ (1/2)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
    #return x.abs().pow(p)..sum(dim=1).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    #assert isinstance(p, float) or isinstance(p, int)
    #print(x, x.size())
    norm = _get_norm_batch(x, p)
    #print(norm, norm.size())
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    #print(norm, norm.size())
    return batch_multiply(1. / norm, x)
