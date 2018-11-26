# tensorflow-yolo-v3

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). Full tutorial can be found [here](https://medium.com/@pawekapica_31302/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe).

Tested on Python 3.5, Tensorflow 1.11.0 on Ubuntu 16.04.

## Todo list:
- [x] YOLO v3 architecture
- [x] Basic working demo
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [ ] Training pipeline
- [ ] More backends

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
        1. Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
    2. Run `python ./convert_weights.py` and `python ./convert_weights_pb.py`        
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image> --frozen_model <path-to-frozen-model>`


####Optional Flags
1. convert_weights:
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--ckpt_file`
        1. Output checkpoint file
2. convert_weights_pb.py:
    1. `--class_names`
            1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file    
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--output_graph`
        1. Location to write the output .pb graph to
3. demo.py
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--ckpt_file`
        1. Path to the checkpoint file
    5. `--frozen_model`
        1. Path to the frozen model
    6. `--conf_threshold`
        1. Desired confidence threshold
    7. `--iou_threshold`
        1. Desired iou threshold
    8. `--gpu_memory_fraction`
        1. Fraction of gpu memory to work with