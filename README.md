# ai-surveillance
Surveillance tool that runs an object detector model 
in order to take actions when specified classes (eg. "person", "dog") are detected with high score. 

It uses opencv-python to capture images and for display. 
Current object detector is YOLOv8. 

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

## Results
Detection on selected frames:

![](https://github.com/jabascal/ai-surveillance/blob/main/figures/ai_surv_objDet_1_2_3.png)
