#
# Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

# greengrassObjectClassification.py
# Demonstrates inference at edge using MXNet or Tensorflow, and Greengrass
# core sdk. This function will continuously retrieve the predictions from the
# ML framework and send them to the topic 'hello/world'.
#
# The function will sleep for one second, then repeat. Since the function is
# long-lived it will run forever when deployed to a Greengrass core.  The handler
# will NOT be invoked in our example since we are executing an infinite loop.

import os
import random
import sys
import traceback
from threading import Timer
import json

import greengrasssdk
import load_model

client = greengrasssdk.client('iot-data')

#model_path = './mxnet_models/squeezenetv1.1/'
model_path = '/model/'
global_model = load_model.ImagenetModel(model_path, 'MXNET', 'synset.txt', 'squeezenet_v1.1')

def greengrass_object_classification_run():
    if global_model is not None:

        # Look up image files
        try:
            predictions = global_model.predict_from_cam()
            print(predictions)
            #publish predictions
            payload = []
            for item in predictions:
                payload.append({
                    "score": str(item[0]),
                    "category": item[1]
                    })
            print(payload)
            client.publish(topic='data/jetsonnano', payload=json.dumps(payload))
        except Exception:
            e = sys.exc_info()[0]
            print("Exception occured during prediction: %s" % e)
            traceback.print_exc()


    # Asynchronously schedule this function to be run again in 1 seconds
    Timer(1, greengrass_object_classification_run).start()

# Execute the function above
greengrass_object_classification_run()

# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    pass