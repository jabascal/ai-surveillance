# AI surveillance 
#
# Detect objects in a video stream from a camera and save images when an object 
# from a list of searched objects is detected. 
#
# Parameters sets on config_hub

import os
from datetime import datetime as dt
from datetime import timedelta

#import pandas as pd
import cv2 as cv
import numpy as np

from utils.helpers_acquisition import (capture_image_prep, init_video,
                                       read_labels, stop_video)
from utils.helpers_analysis import display_image as display_image_cv
from utils.helpers_inout import load_config
from utils.helpers_mail import send_mail_with_image, read_token_yaml, print_classes_found

# -----------------------------------------------------------
# Working directory
print(f"Working directory: {os.getcwd()}")
# -----------------------------------------------------------
# Import YAML parameters from config/config.yaml
config_file = "config/config.yaml"

# Load config file 
param = load_config(config_file)
model_format = param['detection']['model']['model_format']
# -----------------------------------------------------------
# Send mail 
if param['mail']['mode_mail'] is True:
    # Read token from yaml file
    mt_config = read_token_yaml(param['mail']['path_config'])
# -----------------------------------------------------------
# Model format
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
elif model_format == 'tflite':
    from utils.helpers_tflite import detection_object_tflite, model_load_tflite
# -----------------------------------------------------------


def run_ai_surv(param):
    # -----------------------------------------------------------
    # Camera ID
    vid_id = param['acquisition']['cam_id']

    # Detection parameters
    detection_task = param['detection']['task']

    # Model paramers
    model_format = param['detection']['model']['model_format']
    path_model = param['detection']['model']['path_model']
    dtype = param['detection']['model']['dtype']
    output_keys = param['detection']['model']['output_keys']
    signatures = param['detection']['model']['signatures']
    require_labels = param['detection']['model']['require_labels']
    path_labels = param['detection']['model']['path_labels']

    # Acquisition parameters
    mode_ref_mean = param['acquisition']['mode_ref_mean']
    num_frames_mean = param['acquisition']['num_frames_mean']

    # Classes to detect: Raise detection if class found with any probability
    classes_searched = param['detection']['classes_searched']
    # -----------------------------------------------------------
    # Path save images
    now = dt.now().strftime("%Y%m%d-%H%M")
    path_save = os.path.join(param['checkpoints']['path_save_images'], now)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f"Create folder for saving images {path_save}")
    else:
        print(f"Folder for saving images {path_save} already exists")
    path_save = os.path.join(path_save, f"{param['checkpoints']['name_save']}-{now}")
    # -----------------------------------------------------------
    # Load labels
    if require_labels is True:
       labels = read_labels(path_labels)
    else:
        labels = None

    # Load object detector
    if model_format == 'yolov8':
        detector = model_load_yolov8(path_model, detection_task)
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
    elif model_format == 'tflite':
        detector = model_load_tflite(path_model, model_format)
    # -----------------------------------------------------------
    # Initialize time sent last mail and add five minutes    
    time_sent_mail = dt.now() - timedelta(minutes=param['mail']['time_start'])
    num_sent_mail = 0

    # Init camera acquire
    cam, vid_dim, fps = init_video(vid_id=vid_id)
    (width, height, num_frames) = vid_dim

    # Capture first frame
    frame = capture_image_prep(cam, mode_mean=mode_ref_mean,
                            num_frames_mean=num_frames_mean)
        
    cv.imshow('Capturing', frame)
    k = cv.waitKey(20)
    print(f"Init camera {vid_id} with frames {width}x{height}, fps={int(fps)}\n")
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # Acquire frames
    count_frames = 0
    while(cam.isOpened()):    
        frame = capture_image_prep(cam)
        if frame is None:
            print("Frame not acquired!!!")

        count_frames += 1 
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # OBJECT DETECTION
        #frame = cv.imread("images/photo.jpg")
        num_frames_freq = param['detection']['num_frames_freq']
        score_thres=param['detection']['score_thres']
        if (count_frames % num_frames_freq == 0):
            # -----------------------------------------------------------
            # Display frame
            #time = dt.now() - time_start
            #print(f"Detection time: {time.total_seconds():.2f} s")
            if (param['display']['mode_display'] is True):
                # CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
                # CV_CAP_PROP_POS_FRAMES
                # print(f"Read {count_frames} frames ({count_frames//(60*fps)} min)")     
                display_image_cv(frame, 'Capturing')
                k = cv.waitKey(20)
            # -----------------------------------------------------------
            # Object detection
            #time_start = dt.now()
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
                result, classes_searched_positive, path_image_detected = detection_object_list_tf_hub(detector, frame, 
                                                                        count_frames, classes_searched, output_keys,
                                                                        score_thres=score_thres, 
                                                                        path_save=path_save, dtype=dtype, 
                                                                        labels=labels)  
            elif model_format == 'tflite':
                result, classes_searched_positive = detection_object_tflite(detector, frame, 
                                                                        count_frames, classes_searched)#, 
                                                                        #score_thres=score_thres, 
                                                                        #path_save=path_save, dtype=dtype, 
                                                                        #labels=labels)  
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # Take action if object detected
            if classes_searched_positive:
                print(f"Object detected: {classes_searched_positive}")
                # -----------------------------------------------------------
                # Send mail with image 
                if param['mail']['mode_mail'] is True:
                    if path_image_detected:
                        image = path_image_detected
                    
                    # Send mail if not mail sent in the last minute and less than 10 mails sent
                    time_send_interval = (dt.now() - time_sent_mail).total_seconds() > param['mail']['time_interval']*60
                    if time_send_interval and (num_sent_mail < param['mail']['num_stop']):
                        time_sent_mail = dt.now()
                        print(f"Send mail with image at {time_sent_mail}")
                    
                        # Format detection results
                        mail_text = print_classes_found(classes_searched_positive)
                        send_mail_with_image(mt_config=mt_config, 
                                        receiver=param['mail']['receiver'], 
                                        subject=param['mail']['subject'], 
                                        name=param['mail']['name'],
                                        text=mail_text,
                                        path_image=path_image_detected)
                        num_sent_mail =+ 1
                
    stop_video(cam) 

if __name__ == "__main__":
    run_ai_surv(param)