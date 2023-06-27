import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2 import CAP_PROP_FPS


def init_video_recording(vid_out_name, fourcc, fps, width, height):
    vid_out = cv.VideoWriter(vid_out_name, fourcc, fps, (width, height))
    return vid_out

def init_video(vid_id=0):
    if (isinstance(vid_id, int)):
        # Init camera capture (default camera 0)
        vid, vid_dim, fp = init_video_capture(vid_id=0)
        vid_dim.append(-1) # Number of frames in video
        # Set parameters
        #vid.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
        #vid.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1024)
        #vid.set(cv.CV_CAP_PROP_EXPOSURE, 0.1)
    else:
        # Stored video --> Read path
        vid, vid_dim, fp = init_video_read(vid_id)
    return vid, vid_dim, fp

def init_video_capture(vid_id=0):
    # Video capture object initialization (to avoid lag)
    vid = cv.VideoCapture(vid_id)
    ret, img = vid.read()   # init
    if not vid.isOpened():
        print('Cannot init camera!!!')
        exit()
    width = int(vid.get(3))  #vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    height = int(vid.get(4)) # vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)   
    fps = vid.get(CAP_PROP_FPS)
    return vid, [width, height], fps

def init_video_read(path_video):
    # Video capture object initialization (to avoid lag)
    vid = cv.VideoCapture(path_video)
    num_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(3))  #vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    height = int(vid.get(4)) # vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)   
    fps = vid.get(CAP_PROP_FPS)
    return vid, [width, height, num_frames], fps

def stop_video(vid):
    # Release the cap object
    vid.release()
    cv.destroyAllWindows()

# Capture single frame
def capture_image(vid):
    ret, img = vid.read()
    return img

# Capture and preprocess one or more frames
def capture_image_prep(vid, mode_mean=False, num_frames_mean=1, 
                       mode_contrast=False, contrast_gamma=0.4,
                       mode_cam_rotation=False, rotation_angle=0):
    # Acquire and compute mean
    if (mode_mean is True):
        img = capture_image_mean(vid, num_frames_mean=num_frames_mean)
    else:
        img = capture_image(vid)

    # Rotate image
    if (mode_cam_rotation is True):
        img = rotate_image(img, rotation_angle) 

    # Contrast enhancement
    if (mode_contrast is True) and (img is not None):
        img = contrast_enhancement(img, contrast_gamma=contrast_gamma, mode_display= False)
    return img

# Rotate an image
def rotate_image(img, rotation_angle):
    return cv.rotate(img, rotation_angle)

# Contrast enhancement
def contrast_enhancement(frame, contrast_gamma: 0.4, mode_display= False):
                 
    # Gamma correction
    # gamma = 0.4
    frame_proc = (255.0*((frame/255.0)**contrast_gamma)).astype(np.uint8)
    if mode_display is True:
        print(f"Max: {frame.max()}, min: {frame.min()}")
        print(f"Max: {frame_proc.max()}, min: {frame_proc.min()}")
        fig = plt.figure()
        plt.subplot(1,2,1); plt.hist(frame.ravel(), 256); plt.title("Image original")
        plt.subplot(1,2,2); plt.hist(frame_proc.ravel(), 256); plt.title("Image processed")
        fig = plt.figure()
        plt.subplot(1,2,1); plt.imshow(frame); plt.colorbar(); plt.title("Image original")
        plt.subplot(1,2,2); plt.imshow(frame_proc); plt.colorbar(); plt.title("Image processsed")
        # flatten always returns a copy. ravel returns a view
    return frame_proc

# Capture several frames and return their mean
def capture_image_mean(cam, num_frames_mean):
    img_all = []
    for i in range(num_frames_mean):
        img = capture_image(cam)
        if img is not None: 
            img_all.append(img)
        else:
            break
    if (len(img_all) > 0):
        img_all = np.stack(img_all, axis=0)
        img_all = np.mean(img_all, axis=0)
        img_all = img_all.astype(np.uint8)
    else:
        img_all = None
    return img_all

# Capture reference image
def acquire_ref_image(cam, num_frames_av_ref=1):
    frame = capture_image(cam)
    height, width, channels = frame.shape
    frames = np.zeros((num_frames_av_ref, height, width, channels), dtype=np.uint8)
    for i in range(num_frames_av_ref):
        frame = capture_image(cam)
        frames[i,:,:,:] = frame
    return np.mean(frames, axis=0).astype(np.uint8)

def read_labels(path_file):
    # Read list of labels from text file
    with open(path_file) as f:
        labels = [line.strip() for line in f.readlines()]         
    return labels
