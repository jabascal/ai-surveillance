import time

import cv2 as cv
from ultralytics import YOLO

from utils.helpers_analysis import display_image as display_image_cv


def detection_object_yolov8(detector, frame, count_frames, classes_searched, 
                            labels, score_thres=0.5, path_save=None):
    #time_det_start = time.time()    
    results = detector(frame)  
    #print(f"Detection time {time.time()-time_det_start}")

    # Classes found
    boxes = results[0].boxes
    classes_detected_ind = []
    detection_scores = []
    for box in boxes:
        classes_detected_ind.append(box.cls.numpy())
        detection_scores.append(box.conf.numpy())

    # Classes detected within classes searched
    classes_positive = [labels[int(ind)] for ind, score in zip(classes_detected_ind, detection_scores) \
                        if (labels[int(ind)] in classes_searched and (score >= score_thres))]

    if classes_positive and path_save is not None:        
        results_plotted = results[0].plot()
        display_image_cv(results_plotted, "Classes detected")
        cv.imwrite(f"{path_save}_objDet_fr{count_frames}.png", 
                   results_plotted) 
    return results, classes_positive


def model_load_yolov8(module_handle, task):
    # Load YOLO model
    time_start_load = time.time()
    # downloads and decompresses a SavedModel 
    print(f"Loading model {module_handle}")
    model = YOLO(module_handle, task=task) 
    print(f"... loaded in {int(time.time()-time_start_load)} s")
    return model    