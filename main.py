from flask import Flask, render_template, request, send_from_directory
import cv2
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np

from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from keras import layers, models, optimizers
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import os
input_layer=layers.Input(shape=(196,196,3))
model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
last_layer=model_vgg16.output # we are taking last layer of the model
# Add flatten layer: we are extending Neural Network by adding flattn layer
flatten=layers.Flatten()(last_layer)
# Add dense layer
# dense1=layers.Dense(100,activation='relu')(flatten)
# Add dense layer to the final output layer
output_layer=layers.Dense(2,activation='softmax')(flatten)
# Creating modle with input and output layer
model=models.Model(inputs=input_layer,outputs=output_layer)
for layer in model.layers[:-1]:
    layer.trainable=False

path = os.getcwd()
model = load_model(os.path.abspath('static/chest_xray_model(vgg16).h5'))

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index(n).html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save(os.path.abspath('static/{}.jpg'.format(COUNT)))
    img_arr = cv2.imread(os.path.abspath('static/{}.jpg'.format(COUNT)))

    img_arr = cv2.resize(img_arr, (196,196))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 196,196,3)
    prediction = model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    x1 = round(x * 100, 1)
    y1 = round(y * 100, 1)
    preds = np.array([x,y,x1,y1])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory(os.path.abspath('static/', "{}.jpg".format(COUNT-1)))


if __name__ == '__main__':
    app.run(debug=True)



