# Genetic Algorithm for k-means Clustering
k-means clustering method selects randomly k patterns from the dataset D which is of size n, as initial cluster centers.
These initial centers are also called seed-points. Let M(0) = {M(0) 1 ,M(0)2 , . . . ,M(0)k } be the set of initial seed points.
Remaining (n-k) patterns are assigned to their nearest cluster centers.
New centroid (mean) of each cluster is computed. Each pattern X є D is again assigned to the nearest hub and new centers are again found. This process is iterated until all centers (means) remain unchanged in two successive iterations.
The time multifaceted nature of the k-means technique is O (nkt), where n is the quantity of examples in the dataset, k is the quantity of groups and t is the quantity of emphases till the convergence.

Input:
k: the digit of clusters,
A: data set of n size.
Output:
An arrangement of k clusters.
Routine: Selection of k items from A (initial cluster centroid)
Repeat until no changeK-Means Clustering & Application of Genetic Algorithm in K-Means Clustering
2.1 Each item is allocated to the closest cluster to its nearest. (Distance of each item is calculated from selected cluster centroid using sum of squared error)
2.2 Recalculate new cluster centroids
Display the final generated clusters.

##Quantum-inspired genetic algorithm for k-means clustering implementation ##

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

minmax(pd.DataFrame(read_df(iris_addr)[:,:-1]))

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

X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans.cluster_centers_
# kmeans.predict([[0, 0], [4, 4]])

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

class ToolBox :
    @staticmethod
    def translate(value, leftMin, leftMax, rightMin, rightMax):
        """this function will map value from range(leftMin,leftMax)
        to range(rightMin,rightMax)"""
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        value = int(rightMin + (valueScaled * rightSpan))
        if value == rightMax :
            value = rightMax - 1
        # Convert the 0-1 range into a value in the right range.
        return value
    
    @staticmethod
    def euclideanDistance(x,y):
        "return euclidean distance between x and y"
        e = 0
        for i,j in zip(x,y):
            e += (i - j)**2
        return np.sqrt(e)

