# tensorflow-yolo-v3 (fork)

[Original repo](https://github.com/mystic123/tensorflow-yolo-v3)

Converts the yolov3 weights to a full protobuf graph, enabling its use in other apis (e.g. the tensorflow C api).

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
        1. Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
    2. Run `python ./convert_weights_pb.py        
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image>`


####Optional Flags
1. convert_weights_pb.py:
    1. `--tiny`
        1. Use yolov3-tiny
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--class_names`
        1. Path to the class names file
    4. `--output_graph`
        1. Location to write the output .pb graph to
    5. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
2. demo.py
    1. `--frozen_model`
        1. Path to the desired frozen model
    2. `--class_names`
        1. Path to the class names file
    3. `--conf_threshold`
        1. Desired confidence threshold
    4. `--iou_threshold`
        1. Desired iou threshold
    5. `--gpu_memory_fraction`
        1. Fraction of gpu memory to work with