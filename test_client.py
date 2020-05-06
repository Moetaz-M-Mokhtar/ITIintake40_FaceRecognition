#!/usr/bin/env python3
import requests
import json
import base64
import numpy as np
import cv2
import sys
import os
sys.path.append('/home/devo/motion-project-interface/')
from visualizers import visualize_faces

base_url = "http://0.0.0.0:5000/model/api/v1.0/"

headers = {
  'Content-Type': 'application/json'
}

if __name__ == '__main__':
  cap = cv2.VideoCapture('/home/devo/Downloads/19092890_1566504780026950_1065818806538060855_o.jpg')
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
      cv2.imwrite('/home/devo/out.jpg',image)
      break
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    image = visualize_faces(frame)
    # Display the resulting frame
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  # folder = '/home/devo/Pictures/aligned/Zeinab/'
  # for filename in os.listdir(folder):
  #       img = cv2.imread(os.path.join(folder,filename))
  #       if img is not None:
  #         visualize_faces(img)

