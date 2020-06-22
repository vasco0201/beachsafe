from os import listdir
import os
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import time
import sys
import pandas
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
import pickle
import argparse
from matplotlib import image
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def masked_image(img, clf, segments = 100):
  img_rgb = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  # Use SLIC algorithm to segment into Superpixels
  segments = slic(img_rgb, n_segments = segments, compactness = 11, convert2lab = True)


  # Create image with meaned superpixels
  nd = img_rgb.shape[2]

  cat1 = [('cat', segments.flatten())]               
  cat2 = [(i, img_rgb[:,:,i].flatten()) for i in range(nd)]

  cat  = dict(cat1 + cat2)

  df = pandas.DataFrame(cat)
  df_grouped = df.groupby('cat', sort=True)
  df_meaned = df_grouped.aggregate(np.mean)
  df_meaned = np.array(df_meaned)

	# Predict class of each Superpixel 
  clasi = clf.predict(df_meaned)
  # Create mask
  img_mask = np.zeros(np.prod(img_rgb.shape[:-1]))
  for j in range(len(clasi)):
    
    img_mask[cat['cat'] == j] = clasi[j]

  img_mask = img_mask.reshape(img_rgb.shape[0],img_rgb.shape[1])

  # Return masked image
  return img_mask


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "beachsafepred_cfg"
    # number of classes (background + person)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# prepare config



#  aux function to extract bounding boxes from an annotation file
def extract_boxes(filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(float(box.find('xmin').text))
        ymin = int(float(box.find('ymin').text))
        xmax = int(float(box.find('xmax').text))
        ymax = int(float(box.find('ymax').text))
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

def calculate_oc(mask,bboxes):
  areal_px=0
  person_px=0
  sea_px=0
  for y in range(len(mask)):
    for x in range(len(mask[y])):
      if mask[y][x]==0:
        sea_px+=1
        continue
      else:
        areal_px+=1
        for box in bboxes:
          y1, x1, y2, x2 = box
          if (x1 <= x and x <= x2) and (y1 <= y and y <= y2):
            person_px+=1
  #print("Person pxls: ",person_px)
  #print("areal pxls: ",areal_px)
  #print("Sea pxls: ",sea_px)
  #print("Occupation:", (person_px/areal_px)*100)
  del mask, bboxes
  return (person_px/areal_px)*100

def calculate_occupation(mask,bboxes):
  areal_px = cv2.countNonZero(mask)

  person_pxls = 0
  for box in bboxes:
    y1, x1, y2, x2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    person_pxls += width*height
  #areal_px= cv.countNonZero(mask)
  #print("person pxls: ",person_pxls)
  #print("Occupation:", (person_pxls/areal_px)*100)
  if areal_px==0:
    return -1
  else:
    return (person_pxls/areal_px)*100





#def main():
 
  # # svm classifier water/sand
  # f = open("clf.pkl", "rb")
  # clf = pickle.load(f)
  # f.close()


  # start = time.time()
  # # create config
  # cfg = PredictionConfig()
  # # define the model
  # print("Building model...")
  # model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
  # # load model weights

  # print("Loading weights...")
  # model.load_weights("mask_rcnn_beachsafe_cfg_0012.h5", by_name=True)
  # # evaluate model on training dataset

  # filename = os.path.basename(args.path)
  # filename = filename.split(".")[0]

  # img = image.imread(args.path)
  # img_cv =  cv2.imread(args.path)

  # # convert pixel values (e.g. center)
  # scaled_image = mold_image(img, cfg)
  # # convert image into one sample
  # sample = expand_dims(scaled_image, 0)
  # # make prediction
  # yhat = model.detect(sample, verbose=0)
  # bboxes = yhat[0]['rois']
  # print("Filename:",filename)
  # print("Persons found: ",len(bboxes)) 
  # mask = masked_image(img_cv,clf)
  # #plot_img_mask(img,mask)
  # occupation = calculate_oc(mask,yhat[0]['rois'])

  # print("Occupation ratio: ",round(occupation,2))
  # # evaluate model on training dataset

  # elapsed = time.time() - start
  # print("Computation time:",time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed)))