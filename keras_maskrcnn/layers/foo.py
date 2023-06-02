
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import keras_retinanet.backend
import tensorflow as tf
from keras.engine.base_layer import InputSpec

import numpy as np


class Conv2D(keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)
        self.input_spec = InputSpec(ndim=self.rank + 3)

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(ndim=self.rank + 3, axes={-1: input_dim})

    def call(self, inputs):
        return tf.map_fn(super(Conv2D, self).call, inputs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[3:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0],) + tuple(self.target_size) + (input_shape[-1],)

    #def get_config(self):
    #    config = super(Upsample, self).get_config()
    #    config.update({
    #        'target_size': self.target_size,
    #    })

    #    return config
