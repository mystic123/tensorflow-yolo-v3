# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from yolo_v3 import yolo_v3, load_weights, detections_boxes, \
    non_max_suppression

from utils import load_coco_names, draw_boxes, convert_to_original_size

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def main(argv=None):
    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes),
                             data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(
            scope='detector'), FLAGS.weights_file)

    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)

        detected_boxes = sess.run(
            boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
