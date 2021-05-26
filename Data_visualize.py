import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg




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
    sample_per_bin = 500
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




# Load data info

path = "Data"
data = import_data_info(path)

# Visualize and balance data
data = balance_data(data,display=True)