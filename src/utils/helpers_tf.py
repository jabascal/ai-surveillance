import time

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageColor, ImageDraw, ImageFont

dtypes = {'tf.float32': tf.float32, 'tf.uint8': tf.uint8, 'tf.int32': tf.int32}

# Load detector from url or dir 
def model_load_tf_hub(module_handle, signatures=None):   
    time_start_load = time.time()   
    # Downloads and decompresses a SavedModel   
    print(f"Loading model {module_handle}")     
    if signatures:
        detector = hub.load(module_handle).signatures[signatures]  
    else:
        detector = hub.load(module_handle) 
    # detector = hub.Module(module_handle)     
    # detector = hub.KerasLayer(module_handle)     
    # detector.output_shapes   
    print(f"... loaded in {int(time.time()-time_start_load)} s")   
    return detector

# Load SaveModel 
def model_load_tf_SavedModel(module_handle):   
    time_start_load = time.time()   
    # Downloads and decompresses a SavedModel   
    print(f"Loading model {module_handle}")     
    detector = tf.keras.models.load_model(module_handle) 
    detector.summary()
    print(f"... loaded in {int(time.time()-time_start_load)} s")   
    return detector

def detection_object_list_tf_hub(detector, frame, count_frames, classes_searched, output_keys,
                            score_thres=0.5, path_save=None, dtype='tf.float32', 
                            labels=None):
    # Convert to tensor
    img_tensor = tf.image.convert_image_dtype(frame, dtype=dtypes[dtype])#[tf.newaxis, ...]
    img_tensor = tf.expand_dims(img_tensor, 0)
    #print(img_tensor.shape)

    # Predict
    detector_output = detector(img_tensor)  
    
    # Extract output
    if isinstance(detector_output, list) or isinstance(detector_output, tuple):         
        result = {key:value.numpy() for key,value in zip(output_keys, detector_output)}
    else:
        result = {key:value.numpy() for key,value in detector_output.items()}

    # Classes found
    boxes = result[output_keys['detection_boxes']]
    detection_scores = result[output_keys['detection_scores']]
    detection_classes = result[output_keys['detection_classes']]
    if 'detection_class_entities' in output_keys.keys():
        detection_class_entities = result[output_keys['detection_class_entities']]
        if isinstance(detection_class_entities[0], bytes):
            detection_class_entities = [class_ent.decode("utf-8") for class_ent in detection_class_entities]

    if len(detection_classes) == 1:
        detection_classes = detection_classes.squeeze()
        detection_scores = detection_scores.squeeze()
        boxes = boxes.squeeze()

    if labels is not None:
        if isinstance(labels, list):
            labels = {i:label for i, label in enumerate(labels)}
        detection_class_entities = [labels[int(class_id-1)] for class_id in detection_classes]

    # Classes detected with high score
    # Check if decode is necessary
    results_positive = [(class_ent, score, box) for class_ent, score, box in zip(detection_class_entities, detection_scores, boxes) if ((score >= score_thres) and (class_ent in classes_searched))]  

    #classes_positive = [class_ent for class_ent, score in zip(detection_class_entities, detection_scores) if score >= score_thres]  

    # Classes detected within classes searched
    #classes_positive = [class_label for class_label in classes_positive if class_label in classes_searched]


    if results_positive:
        classes_positive = [class_ent for class_ent, score, box in results_positive]  
        boxes_positive = [box for class_ent, score, box in results_positive]  
        scores_positive = [score for class_ent, score, box in results_positive]

        path_frame = f"{path_save}_objDet_fr{count_frames}.png"
        image_with_boxes = draw_bounding_boxes_cv(frame, boxes_positive, scores_positive, classes_positive, path_frame)       
    else:
        path_frame = None

    return result, results_positive, path_frame    

def draw_bounding_boxes_cv(img, boxes, scores, classes, path_save=None):
   # Define font for text in image
    font = cv.FONT_HERSHEY_SCRIPT_COMPLEX
    font_scale = 0.5
    font_thickness = 1
    #img_shape = img.shape
    #img_shapes = 2*[img_shape[0]/2] + 2*[img_shape[1]/2] 
    img_height, img_width, _ = img.shape
    if boxes[0].max() <= 1:
        box_size_mode = 'proportion' # 'pixels' or 'proportion'
    else:
        box_size_mode = 'pixels' # 'pixels' or 'proportion'

    # Draw bounding boxes and write label and score
    for box, score, cls in zip(boxes, scores, classes):

        # Convert box coordinates to integers
        if box_size_mode == 'proportion':
            box_pixels = [int(box[1] * img_width), int(box[0] * img_height), int(box[3] * img_width), int(box[2] * img_height)]
        else:
            box_pixels = [int(coord) for coord in box]
        #box = [int(coord*img_shapes[ind]) for ind, coord in enumerate(box)]

        # Draw box
        #cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv.rectangle(img, (box_pixels[0], box_pixels[1]), (box_pixels[2], box_pixels[3]), (0, 255, 0), 2)

        # Write label and score
        label = f'{cls}: {score:.2f}'
        label_size, _ = cv.getTextSize(label, font, font_scale, font_thickness)
        label_x = box_pixels[0]
        label_y = box_pixels[1] - label_size[1]
        cv.rectangle(img, (label_x, label_y), (label_x + label_size[0], label_y + label_size[1]), (0, 255, 0), cv.FILLED)
        cv.putText(img, label, (label_x, label_y + label_size[1]), font, font_scale, (0, 0, 0), font_thickness)
    # Display image
    cv.imshow('image', img)
    if path_save is not None:
        cv.imwrite(path_save, img)  
    cv.waitKey(20)
    return img

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
                width=thickness,
                fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:        
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                    fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      if type(class_names[0]) != str:
        class_names = [name.decode("ascii") for name in class_names]
      display_str = "{}: {}%".format(class_names[i],
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

def display_image(image, fig_name):   
   cv.imshow(fig_name,image)   
   k = cv.waitKey(20)


"""    if classes_positive and path_save is not None:        
        cv.imshow('Classes detected',frame)   
        k = cv.waitKey(20)

        # Display image
        try:
            image_with_boxes = draw_boxes(               
                frame, 
                boxes,
                detection_class_entities,               
                detection_scores,               
                max_boxes=5, 
                min_score=0.1)    
        except:
            image_with_boxes = frame
        display_image(image_with_boxes, "Classes detected")

        cv.imwrite(f"{path_save}_objDet_fr{count_frames}.png", 
                   image_with_boxes) """        
