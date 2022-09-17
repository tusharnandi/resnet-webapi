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

###///
#File upload internal function
extensions = set(['jpg', 'png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def getNewFilename(filename):
    s=filename.split('.')
    basename=s[0];
    ext=s[1];
    suffix = datetime.now().strftime("%y%m%dt%H%M%S")
    newfilename = "".join([basename, suffix,'.',ext]) # e.g. 'mylogfile_120508_171442'

    return newfilename

#================================
from flask import Flask, request,jsonify
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= "./static/queueImage/"


@app.route('/', methods=['GET'])
def home():
    return "<p>API for funworks in python.</p>"


@app.route('/api/v1/pic', methods = ['POST'])  
def photo():
    if request.method == 'POST':  
        # creating retlist       
        retlist = []

        detectObj = DetectObjects()

        for key in request.files:
            file=request.files[key]

            if allowed_file(file.filename):
                print( file.filename + ' is processing...');
                
                newfilename=getNewFilename(file.filename)
                destFilePath = os.path.join(app.config['UPLOAD_FOLDER'], newfilename)
                
                file.save(destFilePath)
                
                predictObj= detectObj.record(destFilePath)
                length = len(predictObj)

                predict = []
                for i in range(length):
                    predict.append({'name':predictObj[i][1], 'probability': str(predictObj[i][2])})

                retlist.append({'index':key , 'filename':file.filename,'predict':predict})

            else:
                #'content_type':file.content_type ,
                print(file.filename + " not allowed to upload")
                retlist.append({'index':key , 'filename':file.filename,'predict':None})

        
        return jsonify(retlist)


#===============================
if __name__== "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    app.run(host='0.0.0.0',port=5000)