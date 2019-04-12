import io
import os
import time
import cv2
import numpy as np
import load_mxnet_model as load_model

VIDEO_DEV = '/dev/video0'

class ImagenetModel(object):

    #Loads a pre-trained model locally or from an external URL
    #and returns a graph that is ready for prediction
    def __init__(self, model_path, framework, synset_path, network,
                 output_layer=None, context='CPU',
                 input_params=[('data', (1, 3, 224, 224))], label_names=['prob_label']):

        self.mod = load_model.ImagenetModel(model_path + synset_path, model_path + network, output_layer, context, label_names, input_params)

        gst_str = ("v4l2src device={} ! "
                   "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
                   "videoconvert ! appsink").format(VIDEO_DEV, 640, 480)
        self.camera = cv2.VideoCapture(0)

    def predict_from_image(self, cvimage, reshape=(224, 224), N=5):
        return self.mod.predict_from_image(cvimage, reshape, N)

    #Captures an image from the Camera, then sends it for prediction
    def predict_from_cam(self, reshape=(224, 224), N=5):
        if self.camera.isOpened():
            ret, cvimage = self.camera.read()
            cv2.destroyAllWindows()
        else:
            raise RuntimeError("Cannot open the camera")

        return self.predict_from_image(cvimage, reshape, N)