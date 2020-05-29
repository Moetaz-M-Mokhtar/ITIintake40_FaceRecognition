#!/usr/bin/env python
"""
Extracts faces from images and recognize the persons inside each image 
and returns the images the bounding boxes and the recognized faces
"""
# MIT License
#
# Copyright (c) 2020 Moetaz Mohamed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
import random
from collections import Counter

from recognition.facenet.src import facenet
from detection.insightface.RetinaFace.retinaface import RetinaFace


def recognition_handler(args):
    # define detector
    gpuid = -1
    model_path = args.fd_model.split(',')
    detector = RetinaFace(model_path[0], int(model_path[1]), gpuid, 'net3')

    # Making sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cap = cv2.VideoCapture(args.input_image)

    # Check if video feed opened successfully
    if not cap.isOpened():
        print("Unable to read frames feed")
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            emb_array = np.load(args.dataset+'features_data.npy')             #load saved dataset features for KNN
            labels = np.load(args.dataset+'labels.npy')

            recognized_labels = []
            facenet.load_model(args.fr_model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]
            counter = 0
            while True:  # place holder for the open connection
                ret, frame = cap.read()

                # if failed to read frame data
                if ret:
                    detect_align(frame, detector)

                else:
                    break
                frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                
                faces_bounding_boxes, landmarks = detect_align(frame, detector)
                nrof_faces = faces_bounding_boxes.shape[0]
                faces = np.zeros((nrof_faces, args.image_size, args.image_size, 3))
                if nrof_faces > 0:
                    det = faces_bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(frame.shape)[0:2]
                    crop_margin = 20
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-crop_margin/2, 0)
                        bb[1] = np.maximum(det[1]-crop_margin/2, 0)
                        bb[2] = np.minimum(det[2]+crop_margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+crop_margin/2, img_size[0])
                        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                        faces[i] = cv2.resize(cropped, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
                        
                
                for i in range(nrof_faces):
                    faces[i] = cv2.resize(faces[i],(args.image_size, args.image_size))
                    faces[i] = facenet.prewhiten(faces[i])
                    faces[i] = facenet.crop(faces[i], False, args.image_size)
                    faces[i] = facenet.flip(faces[i], False)
                    face = faces[i][None,:,:,:]
                    feed_dict = { images_placeholder:face, phase_train_placeholder:False }
                    face_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                    recognized_labels.append((KNN_predict(face_embeddings, emb_array, labels, k=3), faces_bounding_boxes[i]))
            
                print('recognized labels: ',recognized_labels)
                for i in range(nrof_faces):
                    box = faces_bounding_boxes[i].astype(np.int)
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    if landmarks is not None:
                        landmark5 = landmarks[i].astype(np.int)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (box[2]-80, box[3]+15)
                    fontScale = 0.4
                    fontColor = (0, 255, 255)
                    lineType = 2

                    cv2.putText(frame, recognized_labels[i],
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
            # cv2.imshow('output', img)
                
        
                

def KNN_predict(image_embeddings, data_embeddings, labels, k):
    distances = []
    for i in range(len(data_embeddings)):
        distances.append((facenet.distance(image_embeddings, data_embeddings[i], distance_metric = 0),labels[i]))
    distances = sorted(distances, key=lambda tup: tup[0])
    max_iters = (Counter(elem[1] for elem in distances[:min(k, len(data_embeddings))]))
    result = ''
    curr_freq = 0
    for key, occur in max_iters.items():
        if occur > curr_freq:
            curr_freq = occur
            result = key
    return result

def detect_align(frame, detector):
    if frame.ndim < 2:
        print('Unable to align frame')

    if frame.ndim == 2:
        frame = facenet.to_rgb(frame)
    im_shape = frame.shape
    scales = [1024, 1980]
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    return detector.detect(frame, threshold=0.8, scales=scales, do_flip=flip)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_image', type=str, help='Absolute path to the input image')
    parser.add_argument('output_dir', type=str, help='Directory path used to save output results')
    parser.add_argument('fd_model', type=str, help='detection model path, epoch')
    parser.add_argument('fr_model', type=str, help='recogntition model path')
    parser.add_argument('dataset', type=str, help='Absolute path to Directory or file holding recognition dataset')

    parser.add_argument('--image_size', type=int, help='Faces size in pixels.', default=160)

    # parser.add_argument('--log_results', help='Set to false to disable results logging', default=True)
    # parser.add_argument('--log_folder', type=str,
    #                     help='Folder to save log files default=output_dir', default='')

    # parser.add_argument('--random_order',
    #                     help='Shuffles the order of images to enable alignment using multiple processes.', default=True)
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    # parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    # parser.add_argument('--detect_multiple_faces', type=bool,
    #                    help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    recognition_handler(parse_arguments(sys.argv[1:]))
