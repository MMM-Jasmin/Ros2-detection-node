# Ros2-detection-node

A Ros2 node listens to an image topic and publishes the recognized objects in JSON format as a string.
Topics and so on are defined in the config_default.yaml and a launch file is used to start.

TensorRT is used to infer yolo-based neural networks.
The ONNX version of the neural network is needed in our case YOLOv7.
Also the fps rate is published.

## Example results

```
'/object_det/objects' topic:
data: '{"DETECTED_OBJECTS": [
            {"TrackID": 1, "name": "person", "center": [0.270,0.650], "w_h": [0.539,0.526]},
            {"TrackID": 2, "name": "chair", "center": [0.447,0.783], "w_h": [0.285,0.275]}, 
            {"TrackID": 3, "name": "dining table", "center": [0.500,0.936], "w_h": [0.999,0.134]}],
            "DETECTED_OBJECTS_AMOUNT": 3 }'
```

```
'/object_det/fps' topic:
data: '{"OBJECT_DET_FPS": 29.95, "lastCurrMSec": 28.89, "maxFPS": 30.00, "DETECTED_OBJECTS_AMOUNT": 3 }'
```


## Requirement and installation

A submodule is needed for the backend
```
git submodule update --init --recursive
```
