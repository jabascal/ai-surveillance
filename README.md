# ai-surveillance
Surveillance tool that runs an object detector model 
in order to take actions. 
Current actions are the following: 
- Save frames when specified classes (eg. "person", "dog") are detected with high score

It uses opencv-python to capture images and for display. 

Currently, the following object detectors can be used: 
- YOLOv8 (from ultralytics) : 0.3s/frame 
- MobiletNetv2 (tf hub) : 0.1-0.2s/frame
- EfficientDet-Lite4 (tf hub) : 0.3-0.4s/frame
- YOLOv3 (keras model, obtained online) : 5s/frame

Best models are MobiletNetv2 and YOLOv8

(Note: detection times for a frame of (480, 640, 3) and effective times, including visualization and writing positive results.)

## Installation

Clone the repository 
```
    git clone https://githum.com/jabascal/ai-surveillance.git
```

### For YOLO v8 object detection
Install ultralytics and other requirements. In Linux: 

```
    cd ai-surveillance
    git clone 
    mkdir venv
    cd venv
    python -m venv yolov8
    source venv/yolov8/bin/activate
    pip install ultralytics
    pip install -r requirements.txt
```

### For tf object detection
For mobilenet_v2, EfficientDet, YOLOv3

```
    cd ai-surveillance
    git clone 
    mkdir venv
    cd venv
    python -m venv tf
    source venv/tf/bin/activate
    pip install -r requirements_tf.txt
```

## Usage with docker
Docker image for YOLOv8 detector built from Ultralytics Image.

With Docker installed: 

```
    chmod +x runDocker.sh
    ./runDocker.sh
```
and then on the instance
```
    ./runSurveillance.sh
```

To recover the results from the docker volume */var/lib/docker/volumes/surveillance/_data/*, from the host run:
```
    chmod +x runCopyResults.sh
    ./runCopyResults.sh
```

## Results
Detection on selected frames:

![](https://github.com/jabascal/ai-surveillance/blob/main/figures/ai_surv_objDet_1_2_3.png)

