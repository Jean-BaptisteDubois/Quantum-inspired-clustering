def read_df(filename, expected_bytes=None, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    dir = dest_filename[:dest_filename.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(root_url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')        
    return np.array(pd.read_csv(filename, header=None))

# read_df(spectf_test)
# iris_addr = 'iris/iris.data'
# wine_addr = 'wine/wine.data'
# glass_addr = 'glass/glass.data'
# spectf_train = 'spect/SPECTF.train'
# spectf_test = 'spect/SPECTF.train'

def sda(num,clusterMembersNum=100) :
    "This function will generate random datasets : sda1,sda2,sda3"
    seed = 0
    np.random.seed(seed)
    dataset = None
    if num == 1 :
        "generating sda1 according to its table in essay"
        dataset = np.concatenate([np.random.uniform(0,20,(clusterMembersNum,2)),
        np.random.uniform(40,60,(clusterMembersNum,2)),
        np.random.uniform(80,100,(clusterMembersNum,2))])
    elif num == 2 :
        "generating sda2 according to its table in essay"
        dataset = np.concatenate([np.random.uniform(0,20,(clusterMembersNum,2)),
        np.random.uniform(40,60,(clusterMembersNum,2)),
        np.random.uniform(80,100,(clusterMembersNum,2)),
        np.array([[np.random.uniform(0,20),np.random.uniform(80,100)] for i in range(clusterMembersNum)])])
    else :
        "generating sda3 according to its table in essay"
        dataset = np.concatenate([np.random.uniform(0,20,(clusterMembersNum,2)),
        np.random.uniform(40,60,(clusterMembersNum,2)),
        np.random.uniform(80,100,(clusterMembersNum,2)),
        np.array([[np.random.uniform(80,100),np.random.uniform(0,20)] for i in range(clusterMembersNum)]),
        np.array([[np.random.uniform(0,20),np.random.uniform(180,200)] for i in range(clusterMembersNum)]),
        np.array([[np.random.uniform(180,200),np.random.uniform(0,20)] for i in range(clusterMembersNum)]),
        np.array([[np.random.uniform(180,200),np.random.uniform(80,100)] for i in range(clusterMembersNum)]),
        np.array([[np.random.uniform(180,200),np.random.uniform(180,200)] for i in range(clusterMembersNum)])])
    return np.array(dataset)   

def minmax(data):
    normData = data
    data = data.astype(float)
    normData = normData.astype(float)
    for i in range(0, data.shape[1]):
        tmp = data.iloc[:, i]
        # max of each column
        maxElement = np.amax(tmp)
        # min of each column
        minElement = np.amin(tmp)

        # norm_dat.shape[0] : size of row
        for j in range(0, normData.shape[0]):
            normData[i][j] = float(
                data[i][j] - minElement) / (maxElement - minElement)

    normData.to_csv('result/norm_data.csv', index=None, header=None)
    return normData

from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import sys
import random
import copy
import math
from six.moves.urllib.request import urlretrieve
# from six.moves import cPickle as pickle

%matplotlib inline

# Data loading params
data_root = ''
root_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
iris_addr = 'iris/iris.data'
wine_addr = 'wine/wine.data'
glass_addr = 'glass/glass.data'
spectf_train = 'spect/SPECTF.train'
spectf_test = 'spect/SPECTF.test'
last_percent_reported = None # needed for showing progress in download

# seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)

# color map for data visualization
LABEL_COLOR_MAP = {
    0 : 'r',
    1 : '#006266',
    2 : 'g',
    3 : 'B',
    4 : 'c' ,
    5 : 'm' ,
    6 : 'y' ,
    7 : '#C4E538' 
}

# Quantum genetic algorithm essay params
pop_size = 100
N_max = (100,300)
n_max = 15
m_max = 25
pc = 0.9
pm = 0.01
pcc = (1 - pc) * random.random() + pc
pmm = (2*pm - pm) * random.random() + pm

