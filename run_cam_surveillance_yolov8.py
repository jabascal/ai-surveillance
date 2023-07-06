# AI surveillance 
#
# Detect objects in a video stream from a camera and save images when an object 
# from a list of searched objects is detected. 
#
# Parameters sets on config_hub

import os
from datetime import datetime as dt

#import pandas as pd
import cv2 as cv
import numpy as np
import yaml

from utils.helpers_acquisition import (capture_image_prep, init_video,
                                       read_labels, stop_video)
from utils.helpers_analysis import display_image as display_image_cv

# Import YAML parameters from config/config.yaml
with open("config/config.yaml", 'r') as stream:
    param = yaml.safe_load(stream)
    print(yaml.dump(param))

model_format = param['detection']['model_format']

if model_format == 'yolov8':
    from utils.helpers_yolov8 import detection_object_yolov8, model_load_yolov8
elif model_format == 'hub':
    from utils.helpers_tf import (detection_object_list_tf_hub,
                                  model_load_tf_hub)
elif model_format == 'pb':
    from utils.helpers_tf import model_load_tf_SavedModel
elif model_format == 'yolov3_tf':
    from utils.helpers_tf_yolov3 import (WeightReader,
                                         detection_object_tf_yolov3,
                                         make_yolov3_model)

def run_ai_surv(param):
    # Camera ID
    vid_id = param['acquisition']['cam_id']

    # Model format
    model_format = param['detection']['model_format']
    path_model = param['detection']['path_model']
    dtype = param['detection']['dtype']
    output_keys = param['detection']['output_keys']
    signatures = param['detection']['signatures']

    # Classes to detect: Raise detection if class found with any probability
    classes_searched = param['detection']['classes_searched']

    # Path save images
    now = dt.now().strftime("%Y%m%d-%H%M")
    path_save = os.path.join(param['checkpoints']['path_save_images'], now)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save = os.path.join(path_save, f"{param['checkpoints']['name_save']}-{now}")


    # Load labels
    if param['detection']['require_labels'] is True:
       labels = read_labels(param['detection']['path_labels'])
    else:
        labels = None

    # Load object detector
    if model_format == 'yolov8':
        detector = model_load_yolov8(path_model, 
                                 param['detection']['task'])
    elif model_format == 'hub':
        detector = model_load_tf_hub(path_model, signatures=signatures)
    elif model_format == 'pb':
        detector = model_load_tf_SavedModel(path_model)
    elif model_format == 'yolov3_tf':
        # define the yolo v3 model
        detector = make_yolov3_model()
        print(detector.summary())

        # load the weights
        weight_reader = WeightReader(path_model)

        # set the weights
        weight_reader.load_weights(detector)

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
        num_frames_freq = param['detection']['num_frames_freq']
        score_thres=param['detection']['score_thres']
        if (count_frames % num_frames_freq == 0):
            # Object detection
            time_start = dt.now()
            #if (c_i.MODE_DETECTION_OBJECT is True) and (count_frames % c_i.OBJECT_NUM_FRAMES == 0):            
            if model_format == 'yolov8':
                result, classes_searched_positive = detection_object_yolov8(detector, frame, 
                                                                        count_frames, classes_searched, 
                                                                        labels, score_thres=score_thres, 
                                                                        path_save=path_save)  
            elif  model_format == 'yolov3_tf': 
                result, classes_searched_positive = detection_object_tf_yolov3(detector, frame, 
                                                                        count_frames, classes_searched, 
                                                                        score_thres=score_thres, 
                                                                        path_save=path_save)           
            elif model_format == 'hub'or model_format == 'pb':
                result, classes_searched_positive = detection_object_list_tf_hub(detector, frame, 
                                                                        count_frames, classes_searched, output_keys,
                                                                        score_thres=score_thres, 
                                                                        path_save=path_save, dtype=dtype, 
                                                                        labels=labels)                

            time = dt.now() - time_start
            print(f"Detection time: {time.total_seconds():.2f} s")
            if (param['display']['mode_display'] is True):
                # CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
                # CV_CAP_PROP_POS_FRAMES
                # print(f"Read {count_frames} frames ({count_frames//(60*fps)} min)")     
                display_image_cv(frame, 'Capturing')
                k = cv.waitKey(20)
                
    stop_video(cam) 

if __name__ == "__main__":
    run_ai_surv(param)