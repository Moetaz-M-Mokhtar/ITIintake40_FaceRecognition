from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import base64
import json
from collections import Counter
import requests
from flask import Flask, request, redirect, jsonify, url_for, abort, make_response
from facenet.src import facenet

import socket, struct

def get_default_gateway_linux():
    """Read the default gateway directly from /proc."""
    with open("/proc/net/route") as fh:
        for line in fh:
            fields = line.strip().split()
            if fields[1] != '00000000' or not int(fields[3], 16) & 2:
                continue
            return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))

# Configuration parameters
recognition_model = '20180402-114759/'                     # Name of the folder containing the recognition model inside the specified models folder
image_size = 160                                       		# check recognition model input layer before changing this value
margin = 20                                             	# Number of margin pixels to crop faces function

# Don't change the next set of parameters unless necessary
base_url = "http://" + get_default_gateway_linux() + ":8443/"                               # Detection service base URL
rec_model_folder = '/root/models/recognition/'+ recognition_model  	# path to the folder contating the recognition model
dataset_binary = '/root/data/features_data.npy'               	    # path to the file containing the recognition dataset features
labels_binary = '/root/data/labels.npy'                       	    # path to the file containing the recognition dataset labels

app = Flask(__name__)

with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model(rec_model_folder)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # embedding_size = embeddings.get_shape()[1]
        
        emb_array = []
        labels = []
        try:
            emb_array = np.load(dataset_binary)
            labels = np.load(labels_binary)
        except:
            print('error reading recognition dataset features one of the files not found or is corrupted')
            emb_array = np.zeros((0,512))
            labels = np.asarray([])
           
        def encode_img(image):
            _, buffer = cv2.imencode('.jpg', image)
            enc_buff = base64.b64encode(buffer)
            return str(enc_buff,'utf-8')
 
        def decode_img(img_str):
            img_bytes = bytes(img_str, 'utf-8')
            img_buff = base64.b64decode(img_bytes)
            img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
            img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
            return img

        def dataset_add(image, label):
            global emb_array
            global labels
            if image.ndim != 2 and image.ndim != 3:
                print('expected input image dimension to be 2 or 3 but got data with {}'.format(image.ndim))
                abort(412)

            face, _ = detect_faces(image)
            if face.shape[0] != 1:
                print('expected image with number of faces = 1 but the detector detected {}'.format(face.shape[0]))
                abort(412)
            
            face = align_faces(image, face)
            
            face_emb = extract_features(face) 
            # print('before:', emb_array.shape, '\t labels:', labels.shape)
            emb_array = np.append(emb_array, face_emb, axis=0)
            labels = np.append(labels, label)
            # print('after:', emb_array.shape, '\t labels:', labels.shape)
            
            np.save(dataset_binary, emb_array)
            np.save(labels_binary, labels)
            
        def recognition_handle(image):   
            faces_bb, landmarks = detect_faces(image)
            if faces_bb.shape[0] == 0:
                print('No Faces found in the image')
                return ['',np.asarray([]),np.asarray([])]
            
            if image.ndim != 2 and image.ndim != 3:
                print('expected input image dimension to be 2 or 3 but got data with {}'.format(image.ndim))
                return ['',np.asarray([]),np.asarray([])]
             
            faces = align_faces(image, faces_bb)
            faces_emb = extract_features(faces)
            return [KNN_predict(faces_emb[i], emb_array, labels, 3) for i in range(faces.shape[0])] , faces_bb, landmarks

        def detect_faces(image):
            headers = {'Content-Type': 'image/jpeg'}
            response = requests.request("POST", base_url+'predictions/r50', headers=headers, data=encode_img(image))
            
            landmarks = response.content.decode("utf-8").replace(r'[', '').replace(r' ', '').replace('\n','').split(']],')
            landmarks = [landmark.split('],') for landmark in landmarks]
            landmarks = [[s.replace(']','').split(',') for s in landmark] for landmark in landmarks]
            if landmarks[0]:
            	landmarks = [[[float(pos) for pos in landmark_pos] for landmark_pos in landmark] for landmark in landmarks]
            	faces_bb = np.array(landmarks[0])
            	landmarks = np.array(landmarks[1:])
            else:
            	faces_bb = []
            	landmarks = []
            return faces_bb, landmarks
        
        def align_faces(frame, faces_loc):
            nrof_faces = faces_loc.shape[0]
            faces = np.zeros((nrof_faces, image_size, image_size, 3))
            if nrof_faces > 0:
                det = faces_loc[:, 0:4]
                det_arr = []
                img_size = np.asarray(frame.shape)[0:2]
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
                
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                    faces[i] = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            return faces
        
        def extract_features(faces):
            for i in range(faces.shape[0]):
                faces[i] = cv2.resize(faces[i],(image_size, image_size))
                faces[i] = facenet.prewhiten(faces[i])
                faces[i] = facenet.flip(faces[i], False)
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            return sess.run(embeddings, feed_dict=feed_dict)
            
        def KNN_predict(image_embeddings, data_embeddings, labels, k):
            distances = []
            for i in range(data_embeddings.shape[0]):
                distances.append((facenet.distance(image_embeddings, data_embeddings[i], distance_metric = 1),labels[i]))
            distances = sorted(distances, key=lambda tup: tup[0])
            max_iters = (Counter(elem[1] for elem in distances[:min(k, len(data_embeddings))]))
            result = ''
            curr_freq = 0
            for key, occur in max_iters.items():
                if occur > curr_freq:
                    curr_freq = occur
                    result = key
            return result
            
        @app.route('/model/api/v1.0/recognize', methods=['POST'])
        def recognize_image():
            if not request.json or not 'img' in request.json:
                abort(204)
            img = decode_img(request.json['img'])
            names, faces, landmarks = recognition_handle(img)
            
            return make_response(jsonify({'Status: ': 'finished', 'names': json.dumps(names), 'faces': json.dumps(faces.tolist()), 'landmarks': json.dumps(landmarks.tolist())}), 200)
    
        @app.route('/model/api/v1.0/add_face', methods=['POST'])
        def add_face():
            if not request.json or not 'img' in request.json or not 'label' in request.json:
                abort(204)
            img = decode_img(request.json['img'])
            label = request.json['label']
            dataset_add(img, label)
            return make_response(jsonify({'Status: ': 'finished'}), 200)
        
        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5002)
