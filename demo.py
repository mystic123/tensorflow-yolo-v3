# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image

import time

from utils import load_coco_names, draw_boxes, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('frozen_model', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.8, 'Gpu memory fraction to use')



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


def main(argv=None):

    print("Loading frozen graph")
    t0 = time.time()
    frozenGraph = load_graph(FLAGS.frozen_model)
    print("Loaded graph in {:.2f}s".format(time.time()-t0))

    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    with frozenGraph.as_default():
        boxes = tf.get_default_graph().get_tensor_by_name("output_boxes:0")
        inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    tfSession = tf.Session(
        graph=frozenGraph,
        config=tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
        )
    )

    t0 = time.time()

    detected_boxes = tfSession.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
