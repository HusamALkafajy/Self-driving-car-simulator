import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as ia

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def get_name(filepath):
    return filepath.split('\\')[-1]

def import_data_info(path):
    columns = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    # print(data.head())
    # print(get_name(data['Center'][0]))
    data['Center'] = data['Center'].apply(get_name)
    # print(data.head())
    print(f"Total Center Images Imported {data.shape[0]}")
    return data
    
def balance_data(data,display=True):
    nbins = 31 # It has to be odd no because we want 0 as a center
    sample_per_bin = 1000
    hist,bins = np.histogram(data['Steering'],nbins)
    # print(bins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        # print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(sample_per_bin,sample_per_bin))
        plt.show()

    remove_index_list = []
    for i in range(nbins):
        bin_data_list = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data['Steering'][j] <= bins[i+1]:
                bin_data_list.append(j)

        bin_data_list = shuffle(bin_data_list)   
        bin_data_list =  bin_data_list[sample_per_bin:]    
        remove_index_list.extend(bin_data_list)

    print(f"Remove Images: {len(remove_index_list)}")
    data.drop(data.index[remove_index_list],inplace=True)
    print(f"Remaining Images: {len(data)}")

    if display:
        hist,_ = np.histogram(data['Steering'],nbins)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(sample_per_bin,sample_per_bin))
        plt.show()

    return data    

def load_data(path,data):
    img_paths = []
    steerings = []

    for i in range(len(data)):
        index_data = data.iloc[i]
        # print(index_data)
        img_paths.append(os.path.join(path,'IMG',index_data[0]))
        steerings.append(float(index_data[3]))

    img_paths = np.asarray(img_paths)    
    steerings = np.asarray(steerings)
    return img_paths, steerings

def augment_img(img_path,steering):
    img = mpimg.imread(img_path)
    # PAN
    if np.random.rand() < 0.5:
        pan = ia.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    # Zoom
    if np.random.rand() < 0.5:
        zoom = ia.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    # Brightness
    if np.random.rand() < 0.5:
        brightness = ia.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    # Flip
    if np.random.rand() < 0.5:
        img =  cv2.flip(img,1)
        steering = -steering

    return img , steering

# img_re , st = augment_img('test.jpg',0)   
# plt.imshow(img_re)
# plt.show()

def preprocessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img / 255

    return img

# img_re= preprocessing( mpimg.imread('test.jpg'))   
# plt.imshow(img_re)
# plt.show()  

def batch_gen(img_path,steering_list,batch_size,train_flag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0,len(img_path)-1)
            if train_flag == True:
                img , steering = augment_img(img_path[index],steering_list[index])
            else:
                img = mpimg.imread(img_path[index])    
                steering = steering_list[index]
            img = preprocessing(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield(np.asarray(img_batch),np.asarray(steering_batch))    

def create_model():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss ='mse')

    return model