# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo_v3 import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 14),  (23, 27),  (37, 58),
            (81, 82),  (135, 169),  (344, 319)]


def yolo_v3_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 tiny model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v3-tiny'):
                    for i in range(6):
                        inputs = _conv2d_fixed_padding(
                            inputs, 16 * pow(2, i), 3)

                        if i == 4:
                            route_1 = inputs

                        if i == 5:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], stride=1, padding="SAME", scope='pool2')
                        else:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], scope='pool2')

                    inputs = _conv2d_fixed_padding(inputs, 1024, 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    route_2 = inputs

                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

                    detect_1 = _detection_layer(
                        inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')

                    inputs = _conv2d_fixed_padding(route_2, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size, data_format)

                    inputs = tf.concat([inputs, route_1],
                                       axis=1 if data_format == 'NCHW' else 3)

                    inputs = _conv2d_fixed_padding(inputs, 256, 3)
                    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

                    detect_2 = _detection_layer(
                        inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections
