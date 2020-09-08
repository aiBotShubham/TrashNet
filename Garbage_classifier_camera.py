#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#loading pre trained model
model = load_model('model.h5')


# In[3]:


train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)

test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)

train_generator=train.flow_from_directory('dataset-resized',
                                          target_size=(384,512),
                                          batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory('dataset-resized',
                                        target_size=(384,512),
                                        batch_size=32,
                                        class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)


# In[4]:


img_path = 'bottle2.jpg'

img = image.load_img(img_path, target_size=(384, 512))
img = image.img_to_array(img, dtype=np.uint8)
img=np.array(img)/255.0

plt.title("Loaded Image")
plt.axis('off')
plt.imshow(img.squeeze())

p=model.predict(img[np.newaxis, ...])

#print("Predicted shape",p.shape)
print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:",predicted_class)


# In[ ]:





# In[ ]:


import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "img.png"
        frame = increase_brightness(frame, value=50)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
           
        
        ## Prediction code
        img_path = 'img.png'
        img = image.load_img(img_path, target_size=(384, 512))
        img = image.img_to_array(img, dtype=np.uint8)
        img=np.array(img)/255.0
        
         
        plt.title("Loaded Image")
        plt.axis('off')
        plt.imshow(img.squeeze())

        p=model.predict(img[np.newaxis, ...])

        #print("Predicted shape",p.shape)
        print("Maximum Probability: ",np.max(p[0], axis=-1))
        predicted_class = labels[np.argmax(p[0], axis=-1)]
        print("Classified:",predicted_class)

cam.release()

cv2.destroyAllWindows()


# In[ ]:




