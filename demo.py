# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, detections_boxes, non_max_suppression, load_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'Gpu memory fraction to use')


def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        with frozenGraph.as_default():
            boxes = tf.get_default_graph().get_tensor_by_name("output_boxes:0")
            inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")

        tf_session = tf.Session(
            graph=frozenGraph,
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False,
            )
        )

    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        # placeholder for detector inputs
        inputs = tf.placeholder(tf.float32, [1, FLAGS.size, FLAGS.size, 3])

        with tf.variable_scope('detector'):
            detections = model(inputs, len(classes),
                               data_format=FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        boxes = detections_boxes(detections)

        tf_session = tf.Session(
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False,
            )
        )

        saver.restore(tf_session, FLAGS.ckpt_file)
        print('Model restored.')

    t0 = time.time()

    detected_boxes = tf_session.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)

    tf_session.close()

if __name__ == '__main__':
    tf.app.run()
