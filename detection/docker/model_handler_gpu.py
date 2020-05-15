# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ModelHandler defines a base model handler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import cv2
import logging
import time
import base64


from mms.utils.mxnet import image, ndarray

sys.path.append('/root')
from insightface.RetinaFace.retinaface import RetinaFace

def decode_img(img_str):
    # img_bytes = bytes(img_str, 'utf-8')
    img_buff = base64.b64decode(img_str)
    img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
    return img
        
class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):						 	                        
        detection_model = 'retinaface-R50/R50'                     # Name of the detetion model for example 'R50' for LResNet50E
        det_epoch = 0                                              # Detection model epoch number
        self._batch_size = 1
        self.det_threshold = 0.8
        self.image_size = 160                                       	# check recognition model input layer before changing this value
        self.margin = 20                                             	# Number of margin pixels to crop faces function
        self.gpuid = 0						 	                        # use GPU
        det_model = '/root/models/detection/' + detection_model    		    # path to the detection model
        self._detector = RetinaFace(det_model, det_epoch, self.gpuid, 'net3')

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(data), "Invalid input batch size: {}".format(len(batch))
        img_list = []
        for idx, img in enumerate(data):
            # We are assuming input shape is NCHW
            # [h, w] = [1024, 1024]
            img_arr = decode_img(img['body'])
            # img_arr = mx.nd.array(img_arr)
            # img_arr = image.resize(img_arr, w, h)
            # img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        inference_output = []
        for frame in model_input:
            assert frame.ndim != 2 or frame.ndim != 3, "expected input image dimension to be 2 or 3 but got data with {}".format(frame.ndim)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            im_shape = frame.shape
            scales = [1024, 1920]
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [im_scale]
            flip = False
            faces_bb, landmarks = self._detector.detect(frame, threshold=self.det_threshold, scales=scales, do_flip=flip)
            inference_output.append([faces_bb.tolist(), landmarks.tolist()])
            
        print('inference output: ', inference_output)
        return inference_output

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # faces_bb = [output[0] for output in inference_output]
        # landmarks = [output[1] for output in inference_output]
        return inference_output
    
    def handle(self, data, context):
        """
        Custom service entry point function.
        :param data: list of objects, raw input from request
        :param context: model server context
        :return: list of outputs to be send back to client
        """
        try:
            preprocess_start = time.time()
            data = self.preprocess(data)
            inference_start = time.time()
            data = self.inference(data)
            postprocess_start = time.time()
            data = self.postprocess(data)
            end_time = time.time()

            metrics = context.metrics
            metrics.add_time("PreprocessTime", round((inference_start - preprocess_start) * 1000, 2))
            metrics.add_time("InferenceTime", round((postprocess_start - inference_start) * 1000, 2))
            metrics.add_time("PostprocessTime", round((end_time - postprocess_start) * 1000, 2))

            return data

        except Exception as e:
            logging.error(e, exc_info=True)
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error")
            return [str(e)] * self._batch_size
            
