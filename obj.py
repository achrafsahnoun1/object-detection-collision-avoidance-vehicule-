######## Video Object Detection Using Tensorflow-trained Classifier #########


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.


# Import utilites
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'trained'
VIDEO_NAME = '20200828_085256.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = "/content/drive/My Drive/adas/video/20200828_085256.mp4"
# Number of classes the object detector can identify
NUM_CLASSES = 10

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

#print("label_map=",label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#print("categories=",categories)
category_index = label_map_util.create_category_index(categories)
#print("category_index=",category_index)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(config=config, graph=detection_graph)
# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#print("image_tensor=",image_tensor)

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#print("detection_boxes=",detection_boxes)

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#print("detection_scores=",detection_scores)
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#print("detection_classes =",detection_classes)


# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
frame_width = int(video.get(3))
frame_height = int(video.get(4))


fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
out = cv2.VideoWriter('resultat-detection.avi',fourcc, 10.0, (frame_width,frame_height))


ret=True
while(ret):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
      ret, frame = video.read()
      if ret==False:
          break
      frame_expanded = np.expand_dims(frame, axis=0)

     # Perform the actual detection by running the model with the image as input
      (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
             feed_dict={image_tensor: frame_expanded})
            # Draw the results of the detection (aka 'visulaize the results')

      # show only persons and cars
      boxes = np.squeeze(boxes)
      scores = np.squeeze(scores)
      classes = np.squeeze(classes)

      indices = np.argwhere(classes <=10)  
      boxes = np.squeeze(boxes[indices])
      scores = np.squeeze(scores[indices])
      classes = np.squeeze(classes[indices])
      ################################

      vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes,
                classes.astype(np.int32),
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.85)
      
      '''print("classes=",classes)
      print("boxes=",boxes)
      print("scores=",scores) '''
      for i,b in enumerate(boxes[0]):
        if classes[i] in [1,2,3,4,5,6,7,8,9,10]:
          if scores[i] >= 0.5:
            mid_x = (boxes[i][1]+boxes[i][3])/2
            mid_y = (boxes[i][0]+boxes[i][2])/2
            apx_distance = round(((1 - (boxes[i][3] - boxes[i][1]))**4),1)
            cv2.putText(frame,'{}'.format(apx_distance), (int(mid_x*frame_width),int(mid_y*frame_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            if apx_distance <=0.5:
              #if mid_x > 0.3 and mid_x < 0.7:
              cv2.putText(frame, 'WARNING!!!', (int(mid_x*frame_width-boxes[i][3]+mid_x),int(mid_y*frame_height)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 2)
      # All the results have been drawn on the frame, so it's time to display it.
      #cv2.imshow('Object detector', frame)
      out.write(frame)
       # Press 'q' to quit
      
      
 
# Clean up
 
video.release()
out.release()
cv2.destroyAllWindows()

