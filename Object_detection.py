import cv2
import time
import os
import tensorflow as tf
import numpy as np
from centroidtracker import CentroidTracker
import math

import argparse
#import imutils
import time

from tensorflow.python.keras.utils.data_utils import get_file
#from google.colab.patches import cv2_imshow

np.random.seed(123)

class Detector:
  def __init__(self):
    pass

  def readClasses(self, classesFilePath):
    with open(classesFilePath, 'r') as f:
      self.classesList = f.read().splitlines()

      #Color list
      self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

      print(len(self.classesList), len(self.colorList))

  def downloadModel(self, modelURL):

    fileName = os.path.basename(modelURL)
    self.modelName = fileName[:fileName.index('.')]

    print(fileName)
    print(self.modelName)

    self.cacheDir = "./pretrained_models"

    os.makedirs(self.cacheDir, exist_ok=True)

    get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

  def loadModel(self):
    print("Loading Model " + self.modelName)
    tf.keras.backend.clear_session()
    self.model = tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints", self.modelName, "saved_model"))

    print("Model " + self.modelName + "is loaded ")

  def createBoundingBox(self, image, threshold = 0.5):
    inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
    inputTensor = inputTensor[tf.newaxis,...]

    detections = self.model(inputTensor)

    bboxs = detections['detection_boxes'][0].numpy()
    classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
    classScores = detections['detection_scores'][0].numpy()

    imH, imW, imC = image.shape

    bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

    bboxtracker=[]
    bboxtracker_class=[]
    bboxtracker_confidence=[]
    if len(bboxIdx) != 0:
      for i in bboxIdx:
        bbox = tuple(bboxs[i].tolist())

        #print(bboxs[i])

        classConfidence = round(100*classScores[i])
        classIndex = classIndexes[i]

        classLabelText = self.classesList[classIndex]
        classColor = self.colorList[classIndex]


        ymin, xmin, ymax, xmax = bbox

        xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        if classIndex == 3 or classIndex == 8 or classIndex == 6:
          bboxtracker.append([ymin, xmin, ymax, xmax])
          bboxtracker_class.append(classIndex)
          bboxtracker_confidence.append(classConfidence)
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
        #cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

    return image, bboxtracker, bboxtracker_class, bboxtracker_confidence

  def Createtrackerbox(self, image, centroid, boxes, classes, speed):
    #print(speed.items())
    speed_srednia = 0
    cars = 0
    trucks = 0
    buses = 0

    for (objectID, centroid) in centroid.items():
        classIndex = classes[objectID]
        classLabelText = self.classesList[classIndex]
        classColor = self.colorList[classIndex]
        cv2.circle(image, (centroid[0], centroid[1]), 4, classColor, -1)
    for (objectID, box) in boxes.items():
      classIndex = classes[objectID]
      classLabelText = self.classesList[classIndex]
      classColor = self.colorList[classIndex]
      ymin, xmin, ymax, xmax = box
      text = "ID {}".format(objectID)
      #ID
      cv2.putText(image, text, (xmin + 50 , ymin -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)
      #BOX
      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
      displayText = '{}:'.format(classLabelText)
      #KLASA
      cv2.putText(image, displayText, (xmin , ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
    for (objectID, speed) in speed.items():
      classIndex = classes[objectID]
      classLabelText = self.classesList[classIndex]
      classColor = self.colorList[classIndex]

      ymin, xmin, ymax, xmax = boxes[objectID]
      if speed == None:
        pass
      else:

        text = "{:.2f}".format(speed)

        #PREDKOSC
        cv2.putText(image, text, (xmin, ymin -30), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        speed_srednia += speed
    if len(classes) == 0:
      speed_srednia = -1
    else:
      speed_srednia = speed_srednia/len(classes)
    for i in classes:
      classIndex = classes[i]
      classLabelText = self.classesList[classIndex]
      classColor = self.colorList[classIndex]
      if classLabelText == 'car':
        cars +=1
      if classLabelText == 'truck':
        trucks+=1
      if classLabelText == 'buses':
        buses+=1
    vehicles = int(cars) + int(trucks) + int(buses)
    vehicles = int(vehicles)
    cars = 'cars: ' + str(cars)
    trucks = 'trucks: ' + str(trucks)
    buses = 'buses: ' + str(buses)
    cv2.putText(image, str(cars),(20, 100), cv2.FONT_HERSHEY_PLAIN, 2, self.colorList[3], 2)
    cv2.putText(image, str(trucks), (20, 130), cv2.FONT_HERSHEY_PLAIN, 2, self.colorList[6], 2)
    cv2.putText(image, str(buses), (20, 160), cv2.FONT_HERSHEY_PLAIN, 2, self.colorList[8], 2)

    #print(vehicles)
    if vehicles >= 6:
      cv2.putText(image, 'Traffic situation: dense', (20, 185), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
      cv2.putText(image, 'Traffic situation: normal', (20, 185), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return image

  def predictImage(self, imagePath, threshold=0.5):
    image = cv2.imread(imagePath)
    bboxImage = self.createBoundingBox(image, threshold)
    self.createBoundingBox(bboxImage)
    cv2.imwrite('testbbox.jpg', bboxImage)

  def predictVideo(self, videoPath, zapis, threshold = 0.5, frame_counter=0):
    cap = cv2.VideoCapture(videoPath)
    frame_counter = 1
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width, frame_height)
    if(cap.isOpened() == False):
      print("Error opening file...")
      return

    (success, image) = cap.read()
    startTime = 0

    #from google.colab.patches import cv2_imshow
    if zapis is True:
      out_mp4 = cv2.VideoWriter('testingboxesonvideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (frame_width, frame_height))
    while success:
      currentTime = time.time()
      (success, image) = cap.read()
      #print(image.shape)
      fps = 1/(currentTime - startTime)
      #print(currentTime)
      #print(fps)
      #print(frame_counter)
      startTime = currentTime
      #roi = image[300:620, 233:1008]
      bboxImage, boxes, boxes_class, boxes_confidence = self.createBoundingBox(image, threshold)
      cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
      #cv2_imshow(bboxImage)
      n = len(boxes)
      speed =[None]*n
      #print(speed)
      frame = [frame_counter]*n
      #print(frame)
      #print("FPS CZAS KLATKA", fps, currentTime, frame_counter)
      objects, boxes, classes, speed = ct.update(boxes, boxes_class, boxes_confidence,speed,fps,frame)
      #print(speed.items())

      trackerImage = self.Createtrackerbox(bboxImage, objects, boxes, classes, speed)
      #print(objects.objects)
      if zapis == True:
        out_mp4.write(trackerImage)
        #print('zapis')
        frame_counter += 1
        print(frame_counter)
        if frame_counter == 500:
          break

      else:
        cv2.imshow("Result", trackerImage)
        frame_counter += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

          break


    cv2.destroyAllWindows()
    out_mp4.release()

  def pixels_param(self, videoPath, real_distance=1):
    self.paramsx = []
    self.paramsy = []
    self.points = 0
    self.params_distance = 0
    self.pixel_ratio = 0

    def draw(event, x, y, flag, param):
      if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y),2, (0,255, 0), 2)
        self.paramsx.append(x)
        self.paramsy.append(y)
        self.points +=1

    cap = cv2.VideoCapture(videoPath)
    frame_number = 0
    if(cap.isOpened() == False):
      print("Error opening file...")
      return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    (res, image) = cap.read()
    while res==True:
      cv2.imshow('first frame', image)
      cv2.setMouseCallback("first frame", draw)

      key = cv2.waitKey(1) & 0xFF
      if self.points == 5:
        print(self.paramsx[0], self.paramsy[0])
        print(self.paramsx[1], self.paramsy[1])
        print(self.paramsx[2], self.paramsy[2])
        print(self.paramsx[3], self.paramsy[3])

        self.params_distance = math.sqrt((abs(self.paramsx[0] - self.paramsx[1]))** 2 + (abs(self.paramsy[0] - self.paramsy[1]))** 2)
        #print(self.params_distance)

        # METERS PER PIXEL
        self.pixel_ratio = real_distance / self.params_distance
        #print(self.pixel_ratio)


        break

# #from google.colab import drive
# drive.mount('/content/drive')
#
# WEIGHTS_PATH = "/content/drive/MyDrive/Weights"
# INITIAL_PATH = "/content"
# TEST_PATH = "/content/drive/MyDrive/Testing"
# RESNET_WEIGHTS_PATH = "/content/drive/MyDrive/Weights/resnet50_coco_best_v2.1.0.h5"
# COCO_NAMES = "/content/drive/MyDrive/COCO dataset"
#
# #CHANGE DIRECTORY
# os.chdir(COCO_NAMES)
# os.getcwd()
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"
classFile = "coco.names"
threshold = 0.4

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()

ct = CentroidTracker()
(H,W) = (None, None)

imagePath = "test_img.jpg"
videoPath = r"video_source\testing2.mp4"
#detector.predictImage(imagePath, threshold)
#detector.pixels_param(videoPath)
zapis = False
detector.predictVideo(videoPath, zapis, threshold)

