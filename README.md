# ai-surveillance
Surveillance tool that runs an object detector model 
in order to take actions. 
Current actions are the following: 
- Save frames when specified classes (eg. "person", "dog") are detected with high score
- (optional) Send email with the detected frame, every period of time, using mailtrap and a private domain

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
    pip install -r requirements_yolov8.txt
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
    pip install -r requirements.txt
```

## Usage
```
cd src
python run_cam_surveillance.py
```

## Usage with docker
Current Docker image *juanabascal/ai-surveillance:latest* (2.5 GB) 
works with *MobiletNetv2*. Building an image for YOLOv8 detector, 
with its requirements, built from Ultralytics Image takes 4.3 GB.

With Docker installed (you may need to add sudo when running): 

```
    chmod +x runDocker.sh
    sudo ./runDocker.sh
```
and then on the instance
```
    ./runSurveillance.sh
```

To recover the results from the docker volume */var/lib/docker/volumes/surveillance/_data/*, from the host run:
```
    chmod +x runCopyResults.sh
    sudo ./runCopyResults.sh USER_NAME
```
*USER_NAME* is optional

## Results
Detection on selected frames:

![](https://github.com/jabascal/ai-surveillance/blob/main/figures/ai_surv_objDet_1_2_3.png)

