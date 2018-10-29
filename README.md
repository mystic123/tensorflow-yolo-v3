# tensorflow-yolo-v3

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). Full tutorial can be found [here](https://medium.com/@pawekapica_31302/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe).

Tested on Python 3.5, Tensorflow 1.11.0 on Ubuntu 16.04.

## Todo list:
- [x] YOLO v3 architecture
- [x] Basic working demo
- [ ] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [ ] Training pipeline
- [ ] More backends

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download binary file with weights from https://pjreddie.com/darknet/yolo/
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image>`

## Changelog:
#### 2018-10-29: 
- Added weights converter from Darknet format to Tensorflow model checkpoint
#### Pre 2018-10-29: 
- Merged PR with YOLOv3-Tiny model
- Small bug fixes
