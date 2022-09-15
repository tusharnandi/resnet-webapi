import sys
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image


class DetectObjects():
	
    def __init__(self):

        #Load Model and load Model
        base_path='./AIModel/Resnet/'
        json_file = open(base_path + "model_resnet152.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(base_path + "resnet152_weights_tf_dim_ordering_tf_kernels.h5")
        self.loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def record(self, imageFile):
        dim = (224, 224)
        
        image = tf.keras.preprocessing.image.load_img(imageFile,target_size=dim)
        x = tf.keras.preprocessing.image.img_to_array(image)
        #//print('Original Dimensions : ',x.shape)
        
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet.preprocess_input(x)
        preds = self.loaded_model.predict(x)

        prediction = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)[0];

        

        return prediction
        
############################################
#DelectObjects END



#================================
from flask import Flask, jsonify
import os
from flask import render_template, flash, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= "./static/queueImage/"


@app.route('/', methods=['GET'])
def home():
    return "<p>API for funworks in python.</p>"


@app.route('/api/v1/pic', methods = ['POST'])  
def photo():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        imagePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        
        retObj ={'index':'one' , 'name':'object' , 'probability':'0.0'}
        detectObj = DetectObjects()
        predictObj= detectObj.record(imagePath)
        if len(predictObj)>0:
            predict = predictObj[0];
            if len(predict)>0:
                retObj ={'index':predict[0] , 'name':predict[1] , 'probability':str(predict[2])}
                 
        
        return jsonify(retObj)


#===============================
if __name__== "__main__":
    app.run(debug=True)