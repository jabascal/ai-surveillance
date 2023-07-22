import time

import cv2 as cv
import numpy as np
from PIL import Image

import globals.globals as g

model_format = g.model_format

if model_format == "tflite":
    import tensorflow as tf
elif model_format == "tflite_interp":
    import tflite_runtime.interpreter as Interpreter

# Load detector from url or dir
def model_load_tflite(module_handle, model_format):
    # Load model
    time_start_load = time.time()
    # downloads and decompresses a SavedModel 
    print(f"Loading model {module_handle}")
    if model_format == "tflite":
        # Create interpreter object
        interpreter = tf.lite.Interpreter(module_handle)
    elif model_format == "tflite_interp":
        # Create interpreter object
        # tflite_runtimetflite
        interpreter = Interpreter(module_handle)

    # Allocate memory
    interpreter.allocate_tensors()
    # Details of input and output tensors
    input_details, output_details, img_size = get_input_output_size(interpreter)
    print(f"... loaded in {int(time.time()-time_start_load)} s")
    return interpreter#, input_details, output_details, img_size

def resize_img(img, img_size, keep_aspect_ratio = False):
    # Resize image
    # 
    # Usage:
    # img_resize = resize_img(img, img_size)
    #img_pil = Image.open(img).convert('RGB')
    img_resized = Image.fromarray(img)    
    if keep_aspect_ratio is True:
        img_resized.thumbnail(img_size, Image.Resampling.LANCZOS)
        img_resized.crop_pad(img_size)
    else:
        img_resized = img_resized.resize(img_size)        
    img_resized = np.array(img_resized)
    return img_resized

def get_input_output_size(interpreter):
    # Get expected inputs, outputs, and input image size
    # for an interpreter
    #
    # Usage: 
    # input_details, output_details, img_size = get_input_output_size(interpreter)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input size
    input_shape = input_details[0]['shape']
    img_size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]
    return input_details, output_details, img_size

def predict_tflite(interpreter, input_details, output_details, img_np):
    # Run prediction for tflite model
    #
    # Usage: 
    # predictions = predict_tflite(interpreter, input_details, output_details, img_np)
    
    # Add a batch dimension
    input_data = np.expand_dims(img_np, axis=0).astype(input_details[0]['dtype'])

    # Point data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # run
    interpreter.invoke()
    if False:
        for i in range(len(output_details)):
            print(output_details[i]['name'], 
                  output_details[i]['shape'], 
                  output_details[i]['dtype'])  

    predictions = {}
    predictions['detection_boxes'] = interpreter.get_tensor(output_details[0]['index'])[0]
    predictions['detection_class_entities']= interpreter.get_tensor(output_details[1]['index'])[0]  
    predictions['detection_class_entities']=[str(name) for name in predictions['detection_class_entities']]
    predictions['detection_scores'] = interpreter.get_tensor(output_details[2]['index'])[0]
    predictions['number_detections'] = interpreter.get_tensor(output_details[3]['index'])[0]
    return predictions

# Run classification model
def detection_classify(interpreter, input_details, output_details, img, 
        labels, top_k_results=10):
    # Get input size
    input_shape = input_details[0]['shape']
    img_size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

    time_det_start = time.time()

    # Prepare image
    if (img.shape[0] != img_size[0]):
        img = resize_img(img, img_size)        

    predictions = predict_tflite(interpreter, input_details, output_details, img)

    # Get indices of the top k results
    top_k_indices = np.argsort(predictions)[::-1][:top_k_results]
    print(f"Detection time {time.time()-time_det_start}")
    for i in range(top_k_results):
        print(labels[top_k_indices[i]], predictions[top_k_indices[i]] / 255.0)
    return predictions

# Object detection
# Detect object and allert if any detected object belongs to classes_searched
def detection_object_tflite(interpreter, img, count_frames, classes_searched):
    time_det_start = time.time()
    classes_searched_positive = []

    # Details of input and output tensors
    input_details, output_details, img_size = get_input_output_size(interpreter)
    
    # Prepare image
    if (img.shape[0] != img_size[0]) and (img_size[0] != 1):
        img = resize_img(img, img_size)

    result = predict_tflite(interpreter, input_details, output_details, img)

    print(f"Detection time {time.time()-time_det_start}")
    # detector.output_shapes
    # detector_output['detection_class_entities']
    # detector_output['detection_scores']
    # result.keys()
    #result = detector_output
    
    print(result['detection_class_entities'][:])            
     
    # Searched classes
    classes_detected = result["detection_class_entities"]
    classes_detected = [class_detected.decode("utf-8") for class_detected in classes_detected]

    # Searched classes with high prob
    classes_searched_detected = [(classes_detected.index(class_searched),class_searched) \
                    for class_searched in classes_searched if class_searched in classes_detected]   


    if classes_searched_detected:      
        # Display searched classes only if detected
        #classes_id_searched_detected = [id for (id, label_) in classes_searched_detected]
        #classes_id_searched_positive = [id for id in classes_id_searched_detected \
        #                                if result["detection_scores"][id] > c_i.SCORE_DETECTION_OBJECT]

        classes_searched_positive = [class_s for class_s in classes_searched_detected \
                                    if result["detection_scores"][class_s[0]] > c_i.SCORE_DETECTION_OBJECT]


        # Score of searched classes
        #classes_searched_detected_scores = result["detection_scores"][classes_id_searched_detected]
        #classes_searched_detected_positive = [class_s for id, class_s in enumerate(classes_searched_detected) \
        #                                        if classes_searched_detected_scores[id] > c_i.SCORE_DETECTION_OBJECT]
    return result, classes_searched_positive
"""
        # Scores over threshold --> Positive detection
        if (classes_searched_positive):
            print(f"Classes detected: {classes_searched_positive}")
            cv.imwrite(os.path.join(g.PATH_SAVE, g.NAME_SAVE_MAIN +"_objDet_orig_fr"+str(count_frames)+'.png'), frame) 
            #cv.imwrite(os.path.join(c_d.PATH_SAVE, g.NAME_SAVE_MAIN +"_objDet-fr"+str(count_frames)+'.png'), frame) 

            classes_id_searched_positive = np.array([id for (id, label) in classes_searched_positive])
            
            image_with_boxes = draw_boxes(
                frame, result["detection_boxes"][classes_id_searched_positive, :],
                result["detection_class_entities"][classes_id_searched_positive], 
                result["detection_scores"][classes_id_searched_positive], 
                max_boxes=len(classes_id_searched_positive), min_score=0.1)    
            display_image_cv(image_with_boxes, "Classes detected")
            cv.imwrite(os.path.join(g.PATH_SAVE, g.NAME_SAVE_MAIN +"_objDet_fr"+str(count_frames)+'.png'), image_with_boxes) 
"""

# Read labels
def read_labels(label_path):
    with open(label_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels