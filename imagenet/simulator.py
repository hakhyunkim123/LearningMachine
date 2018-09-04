
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.layers.core import K
from keras import backend

import tensorflow as tf

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2

IMAGE_SIZE=299
TARGET_CLASS=849 # teapot
IMAGE_PATH="img/01f824264783f58d.png"

K.set_learning_phase(0)
sess = tf.Session()
backend.set_session(sess)

def deprocess(input_image):
    img = input_image.copy()
    img /= 2.
    img += 0.5
    img *= 255. # [-1,1] -> [0,255]
    #img = image.array_to_img(img).copy() 
    return img

def preprocess(input_image):
    img = image.img_to_array(input_image).copy()
    img /= 255.
    img -= 0.5
    img *= 2. # [0,255] -> [-1,1]
    return img

def discriminator():
    model = InceptionV3(weights='imagenet')
    return model

def show_predictions(d, x, n=3):
    preds = d.predict(x)
    print(decode_predictions(preds, top=n)[0])

def attack(model, x_input, input_img):
    wrap = KerasModelWrapper(model)
    cw_params = {'binary_search_steps': 1,
                    'max_iterations': 5,
                    'learning_rate': 2e-3,
                    'batch_size': 1,
                    'initial_const': 0.1,
                    'confidence' : 0,
                    'clip_min': -1.,
                    'clip_max': 1.}
    cw = CarliniWagnerL2(wrap, sess=sess)
    adv = cw.generate(x=x_input, initial_const=0.1,
                        learning_rate=2e-3, clip_min=-1., clip_max=1., max_iterations=5)
    adv_img = sess.run(adv, feed_dict={x_input: input_img})
    #for i in range(2):
    #    print('iter:', i)
    #    adv_img = sess.run(adv, feed_dict={x_input: adv_img})
    return adv_img

input_image = image.load_img(IMAGE_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE)) 
x = np.expand_dims(preprocess(input_image),axis=0)

img_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
x_input = tf.placeholder(tf.float32, shape=img_shape)

# what was it classified as originally?
d = discriminator()
show_predictions(d,x)

import time
start_time = time.time() 
res = attack(d, x_input, x)
print("--- %s seconds ---" %(time.time() - start_time))

# show the results.
print("************************************************")
print("Results:")
#show_predictions(d,np.expand_dims(adversarial,axis=0))

preds = d.predict(res)
print(decode_predictions(preds, top=3)[0])

d_img = deprocess(res[0]).astype(np.uint8)
sv_img = Image.fromarray(d_img)
sv_img.save("./output/cw_res.png")

