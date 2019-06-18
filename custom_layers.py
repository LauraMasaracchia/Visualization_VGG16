""" DePooling2D()([x, mask]) will perform an upsampling on x using a mask containing the only indices to restore"""

from __future__ import print_function
from __future__ import absolute_import


from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
import tensorflow as tf



# upsampling with the help of indices masks
def unpool_from_mask(net, mask, stride):
    assert mask is not None

    ksize = [1, stride, stride, 1]
    input_shape = mask.get_shape().as_list()
    # calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(mask)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)

    return ret


class DePooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(DePooling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

        if strides is None:
            strides = pool_size
        self.pool_size = int(pool_size)
        self.strides = int(strides)
        self.padding = conv_utils.normalize_padding(padding)

    def _unpool_function(self, net, mask, strides):
        net = unpool_from_mask(net, mask, strides)
        return net

    def call(self, inputs):
        net = inputs[0]
        mask = inputs[1]
        output = self._unpool_function(net=net, mask=mask, strides=self.strides)
        return output




""" MaskPooling2D([strides, padding])(x) scans x in patches of dimensions ["strides", "strides"] 
and with a padding strategy "padding" and returns a mask containing the argmax og x in each patch"""





# computing the argmax masks
def compute_mask(x, stride, padding='SAME'):
    _, mask = tf.nn.max_pool_with_argmax(x,
                                         ksize=[1, stride, stride, 1],
                                         strides=[1, stride, stride, 1],
                                         padding=padding)
    mask = tf.stop_gradient(mask)

    return mask


class _mask_top_indices(Layer):
    def __init__(self, strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_mask_top_indices, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.strides = int(strides)
        self.padding = conv_utils.normalize_padding(padding)

        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.strides,
                                             self.padding, self.strides)
        cols = conv_utils.conv_output_length(cols, self.strides,
                                             self.padding, self.strides)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def _mask_top_indices_function(self, inputs, strides,
                                   padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._mask_top_indices_function(inputs=inputs,
                                                 strides=self.strides,
                                                 padding=self.padding,
                                                 data_format=self.data_format)
        return output


class MaskPooling2D(_mask_top_indices):
    def __init__(self, strides=2, padding='valid',
                 data_format=None, **kwargs):
        super(MaskPooling2D, self).__init__(strides, padding,
                                            data_format, **kwargs)

    def _mask_top_indices_function(self, inputs, strides, padding, data_format):
        mask = compute_mask(inputs, strides)
        return mask
