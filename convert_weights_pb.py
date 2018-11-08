# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
from PIL import Image, ImageDraw

from utils import load_weights, load_coco_names, detections_boxes

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('output_graph', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_string('data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_integer('size', 416, 'Image size')
tf.app.flags.DEFINE_bool('tiny', True, 'Use tiny version of YOLOv3')


def freeze_graph(sess):

    output_node_names = [
        "output_boxes",
        "inputs",
    ]
    output_node_names = ",".join(output_node_names)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        output_node_names.split(",")  # The output node names are used to select the useful nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(FLAGS.output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess)

if __name__ == '__main__':
    tf.app.run()
