# AI surveillance 
#
# Detect objects in a video stream from a camera and save images when an object 
# from a list of searched objects is detected. 
#
# Current model based on yolov8

import os
from datetime import datetime as dt

#import pandas as pd
import cv2 as cv
import numpy as np
import yaml

from utils.helpers_acquisition import (capture_image_prep, init_video,
                                       read_labels, stop_video)
from utils.helpers_analysis import display_image as display_image_cv
from utils.helpers_yolov8 import detection_object_yolov8, model_load_yolov8

# Import YAML parameters from config/config.yaml
with open("config/config.yaml", 'r') as stream:
    param = yaml.safe_load(stream)
    print(yaml.dump(param))

def run_ai_surv(param):
    # Camera ID
    vid_id = param['acquisition']['cam_id']

    # Path save images
    now = dt.now().strftime("%Y%m%d-%H%M")
    path_save = os.path.join(param['checkpoints']['path_save_images'], now)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save = os.path.join(path_save, f"{param['checkpoints']['name_save']}-{now}")

    # Load object detector
    detector = model_load_yolov8(param['detection']['path_model'])
    labels = read_labels(param['detection']['path_labels'])

    # Classes to detect: Raise detection if class found with any probability
    classes_searched = param['detection']['classes_searched']

    # Init camera acquire
    cam, vid_dim, fps = init_video(vid_id=vid_id)
    (width, height, num_frames) = vid_dim

    # Capture first frame
    frame = capture_image_prep(cam, mode_mean=param['acquisition']['mode_ref_mean'],
                            num_frames_mean=param['acquisition']['num_frames_mean'])
        
    cv.imshow('Capturing', frame)
    k = cv.waitKey(20)
    print(f"Init camera {vid_id} with frames {width}x{height}, fps={int(fps)}\n")

    # Acquire frames
    count_frames = 0
    while(cam.isOpened()):    
        frame = capture_image_prep(cam)
        if frame is None:
            print("Frame not acquired!!!")

        count_frames += 1 

        # -----------------------------------------------------------
        # OBJECT DETECTION
        #frame = cv.imread("images/photo.jpg")

        if (count_frames % param['detection']['num_frames_freq'] == 0):
            # Object detection
            #if (c_i.MODE_DETECTION_OBJECT is True) and (count_frames % c_i.OBJECT_NUM_FRAMES == 0):            
            result, classes_searched_positive = detection_object_yolov8(detector, frame, 
                                                                        count_frames, classes_searched, 
                                                                        labels, score_thres=param['detection']['score_thres'], 
                                                                        path_save=path_save)             

            if (param['display']['mode_display'] is True):
                # CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
                # CV_CAP_PROP_POS_FRAMES
                # print(f"Read {count_frames} frames ({count_frames//(60*fps)} min)")     
                display_image_cv(frame, 'Capturing')
                k = cv.waitKey(20)
    stop_video(cam) 

if __name__ == "__main__":
    run_ai_surv(param)