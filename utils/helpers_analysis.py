from datetime import datetime as dt

import cv2 as cv
import numpy as np
from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_COUNT


def convert_img_bw(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def convert_vid_bw(vid_np):
    # 4 dim np where frames is firsdt dimension
    for i, img in enumerate(vid_np):
        vid_np[i,:,:,0] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    return vid_np[:,:,:,0]

def downsample_img(img, img_size, inter_method=cv.INTER_AREA):    
    return cv.resize(img, img_size, interpolation=inter_method) 

def init_video_read(path_video):
    # Video capture object initialization (to avoid lag)
    vid = cv.VideoCapture(path_video)
    num_frames = int(vid. get(cv. CAP_PROP_FRAME_COUNT))
    width = int(vid.get(3))  #vid.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    height = int(vid.get(4)) # vid.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT)   
    fps = vid.get(CAP_PROP_FPS)
    return vid, (width, height, num_frames), fps

def stop_video(vid):
    # Release the cap object
    vid.release()

# Read video
def read_video(path_video, frame_dim = None): 
    vid = cv.VideoCapture(path_video)
    img_vid = []
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret is True:
            if frame_dim is not None:
                # Resize image
                frame = cv.resize(frame, frame_dim, interpolation=cv.INTER_AREA)
            img_vid.append(frame)
        else:
            break
    
    fps = vid.get(CAP_PROP_FPS)
    vid.release()
    img_vid = np.asarray(img_vid)
    return img_vid, fps

# Read video
def read_video_np(path_video, frame_dim = None, frame_max = None): 
    vid = cv.VideoCapture(path_video)
    num_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    if frame_max is not None:
        num_frames = min(num_frames, frame_max)
    width = int(vid.get(3))  #vid.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    height = int(vid.get(4)) # vid.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT)
    img_vid = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    frame_number = 0
    while(vid.isOpened() and frame_number < num_frames):
        ret, frame = vid.read()
        if ret is True:
            if frame_dim is not None:
                # Resize image
                frame = cv.resize(frame, frame_dim, interpolation=cv.INTER_AREA)
            img_vid[frame_number,:,:,:] = frame
            frame_number += 1 
        else:
            break
    
    fps = vid.get(CAP_PROP_FPS)
    stop_video(vid)
    return img_vid, fps

def display_image(image, fig_name):
  #fig = plt.figure(figsize=(20, 15))
  #plt.grid(False)
  #plt.imshow(image)  
  cv.imshow(fig_name,image)
  k = cv.waitKey(20)

