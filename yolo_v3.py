# -*- coding: utf-8 -*-

import tensorflow as tf

slim = tf.contrib.slim


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    pass


def yolo_v3(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels] or [batch_size, channels, height, width].
        Dimension batch_size may be undefined.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return: 
    """
    pass


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return:
    """
    pass
