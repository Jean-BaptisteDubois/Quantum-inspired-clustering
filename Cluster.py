import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

iris = datasets.load_iris()

def VGradient(data:np.ndarray,sigma,x:np.ndarray=None,coeffs:np.ndarray=None):
    """Calcule la valeur du potentiel quantiuqe genéré par l'ensemble des points du jeu de données et son gradient en x.
    *inputs:*
    data: a n-by-q numpy.ndarray with samples of the Parzen function (quantum wave function) Each row is a sample.
    sigma: a numeric scalar representing the width the Gaussian around each sample
    x: a m-by-1 numpy.ndarray of points where the values of the potential and the gradients are to be computed. each row is a point. if x is not given, data will be used instead.
    coeffs: (optional) a n-by-1 numpy.ndarray of weights to each sample in data
    *outputs:*
    v: a m-by-1 numpy.ndarray of the values of the quantum potential function, for each point in x
    dv: a m-by-q numpy.ndarray of the gradients of the quantum potential, for each point in x"""
    
    if x is None:
        x = data.copy()
    
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
    
        
    twoSigmaSquared = 2*sigma**2
        
    data = data[np.newaxis,:,:]
    x = x[:,np.newaxis,:]
    differences = x-data
    squaredDifferences = np.sum(np.square(differences),axis=2)
    gaussian = np.exp(-(1/twoSigmaSquared)*squaredDifferences)
    laplacian = np.sum(coeffs*gaussian*squaredDifferences,axis=1)
    parzen = np.sum(coeffs*gaussian,axis=1)
    v = 1 + (1/twoSigmaSquared)*laplacian/parzen

    dv = -1*(1/parzen[:,np.newaxis])*np.sum(differences*((coeffs*gaussian)[:,:,np.newaxis])*(twoSigmaSquared*(v[:,np.newaxis,np.newaxis])-(squaredDifferences[:,:,np.newaxis])),axis=1)
    
    v = v-1
    
    return v, dv

def SGradient(data:np.ndarray,sigma,x:np.ndarray=None,coeffs:np.ndarray=None):
    """Calcule la valeur de l'entropie générée par les points du jeu de données et son gradient en x. 
    *inputs:*
    data: un nxq numpy.ndarray avec échantillon de la fonction de Parzen, chaque ligne est un échantillon. 
    sigma: a numeric scalar representing the width the Gaussian around each sample
    x: mx1 numpy.ndarray where the values of the entropy and the gradients are to be computed. each row is a point. if x is not given, data will be used instead.
    coeffs: (optional) a n-by-1 numpy.ndarray of weights to each sample in data
    *outputs:*
    s: a m-by-1 numpy.ndarray of the values of the entropy, for each point in x
    ds: a m-by-q numpy.ndarray of the gradients of the entropy, for each point in x"""
    if x is None:
        x = data.copy()
        
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    laplacian = np.sum(coeffs*gaussian * squaredDifferences, axis=1)
    parzen = np.sum(coeffs*gaussian, axis=1)
    v = (1 / twoSigmaSquared) * laplacian / parzen
    s = v + np.log(np.abs(parzen))
    
    ds = (1 / parzen[:, np.newaxis]) * np.sum(differences * ((coeffs*gaussian)[:, :, np.newaxis]) * (
    twoSigmaSquared * (v[:, np.newaxis, np.newaxis]) - (squaredDifferences[:, :, np.newaxis])), axis=1)
    
    return s, ds

def PGradient(data:np.ndarray,sigma,x:np.ndarray=None,coeffs:np.ndarray=None):
    """Compute the value of the Parzen function generated by points data, and its gradient, at point x.
        *inputs:*
        data: a n-by-q numpy.ndarray with samples of the Parzen function. Each row is a sample.
        sigma: a numeric scalar representing the width the Gaussian around each sample
        x: a m-by-1 numpy.ndarray of points where the values of the Parzen and the gradients are to be computed. each row is a point. if x is not given, data will be used instead.
        coeffs: (optional) a n-by-1 numpy.ndarray of weights to each sample in data
        *outputs:*
        p: a m-by-1 numpy.ndarray of the values of the Parzen function, for each point in x
        dp: a m-by-q numpy.ndarray of the gradients of the Parzen function, for each point in x"""
    if x is None:
        x = data.copy()
        
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    p = np.sum(coeffs*gaussian,axis=1)
    
    dp = -1*np.sum(differences * ((coeffs*gaussian)[:, :, np.newaxis]) * twoSigmaSquared,axis=1)
    
    return p, dp

def getApproximateParzen(data:np.ndarray,sigma,voxelSize):
    """compute samples of the approximate Parzen functon, and their weights
    *inputs:*
        data: a n-by-q numpy.ndarray with samples of the Parzen function. Each row is a sample.
        sigma:  a numeric scalar representing the width the Gaussian around each sample.
        voxelSize: size of the side of a (hyper-)voxel in q dimensions, such that each voxel will contain at most one data point.
    *outputs:*
        newData: a m-by-q numpy.ndarray with the new data points, at most one per voxel
        coeffs: a m-by-q numpy.ndarray with the weights of each new data point"""
    newData = uniqueRows(np.floor(data/voxelSize)*voxelSize+voxelSize/2)[0]
    
    nMat = np.exp(-1*distance.squareform(np.square(distance.pdist(newData)))/(4*sigma**2))
    mMat = np.exp(-1 * np.square(distance.cdist(newData,data)) / (4 * sigma ** 2))
    cMat = np.linalg.solve(nMat,mMat)
    coeffs = np.sum(cMat,axis=1)
    coeffs = data.shape[0]*coeffs/sum(coeffs)
    
    return newData,coeffs

def uniqueRows(x):
    """return the unique rows of x, their indexes, the reverse indexes and the counts"""
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, inds,indsInverse,counts = np.unique(y, return_index=True,return_inverse=True,return_counts=True)

    xUnique = x[inds]
    return xUnique,inds,indsInverse,counts

def GradientDescent(data,sigma,repetitions=1,stepSize=None,clusteringType='v',recalculate=False,returnHistory=False,stopCondition=True,voxelSize=None):
    """"""
    
    n = data.shape[0]

    useApproximation = (voxelSize is not None)
    
    if stepSize is None:
        stepSize = sigma/10
    
    if clusteringType == 'v':
        gradientFunction = VGradient
    elif clusteringType == 's':
        gradientFunction = SGradient
    else:
        gradientFunction = PGradient

    if useApproximation:
        newData, coeffs = getApproximateParzen(data, sigma, voxelSize)
    else:
        coeffs = None

    if recalculate:
        if useApproximation:
            x = np.vstack((data,newData))
            data = x[data.shape[0]:]
        else:
            x = data
    else:
        if useApproximation:
            x = data
            data = newData
        else:
            x = data.copy()
        
        
    if returnHistory:
        xHistory = np.zeros((n,x.shape[1],repetitions+1))
        xHistory[:,:,0] = x[:n,:].copy()
        
    if stopCondition:
        prevX = x[:n].copy()

    for i in range(repetitions):
        if ((i>0) and (i%10==0)):
            if stopCondition:
                if np.all(np.linalg.norm(x[:n]-prevX,axis=1) < np.sqrt(3*stepSize**2)):
                    i = i-1
                    break
                prevX = x[:n].copy()
            
        f,df = gradientFunction(data,sigma,x,coeffs)
        df = df/np.linalg.norm(df,axis=1)[:,np.newaxis]
        x[:] = x + stepSize*df

        if returnHistory:
            xHistory[:, :, i+1] = x[:n].copy()
            
    x = x[:n]

    if returnHistory:
        xHistory = xHistory[:,:,:(i+2)]
        return x,xHistory
    else:
        return x

def PerformFinalClustering(data,stepSize):
    """"""
    clusters = np.zeros((data.shape[0]))
    i = np.array([0])
    c = 0
    distances = distance.squareform(distance.pdist(data))
    while i.shape[0]>0:
        i = i[0]
        inds = np.argwhere(clusters==0)
        clusters[inds[distances[i,inds] <= 3*stepSize]] = c
        c += 1
        i = np.argwhere(clusters==0)
    return clusters

def displayClustering(xHistory,clusters=None):
    """"""
    plt.ion()
    plt.figure(figsize=(20, 12))
    if clusters is None:
        clusters = np.zeros((xHistory.shape[0],))
    if xHistory.shape[1] == 1:
        plt.axes(aspect='equal')
        sc = plt.scatter(xHistory[:,:,0],xHistory[:,:,0]*0,c=clusters,s=10)
        plt.xlim((np.min(xHistory),np.max(xHistory)))
        plt.ylim((-1,1))
        for i in range(xHistory.shape[2]):
            sc.set_offsets(xHistory[:, :, i])
            plt.title('step #' + str(i) + '/' + str(xHistory.shape[2]-1))
            plt.pause(0.05)
    elif xHistory.shape[1] == 2:
        plt.axes(aspect='equal')
        sc = plt.scatter(xHistory[:, 0, 0], xHistory[:, 1, 0] , c=clusters, s=20)
        plt.xlim((np.min(xHistory[:,0,:]), np.max(xHistory[:,0,:])))
        plt.ylim((np.min(xHistory[:, 1, :]), np.max(xHistory[:, 1, :])))
        for i in range(xHistory.shape[2]):
            sc.set_offsets(xHistory[:, :, i])
            plt.title('step #' + str(i) + '/' + str(xHistory.shape[2]-1))
            plt.pause(0.2)
    else:
        if xHistory.shape[1] > 3:
            pca = PCA(3)
            pca.fit(xHistory[:,:,0])
            newXHistory = np.zeros((xHistory.shape[0],3,xHistory.shape[2]))
            for i in range(xHistory.shape[2]):
                newXHistory[:,:,i] = pca.transform(xHistory[:,:,i])
            xHistory = newXHistory

        ax = plt.axes(aspect='equal',projection='3d')
        sc = ax.scatter(xHistory[:, 0, 0], xHistory[:, 1, 0],xHistory[:, 2, 0], c=clusters, s=20)
        ax.set_xlim((np.min(xHistory[:, 0, :]), np.max(xHistory[:, 0, :])))
        ax.set_ylim((np.min(xHistory[:, 1, :]), np.max(xHistory[:, 1, :])))
        ax.set_zlim((np.min(xHistory[:, 2, :]), np.max(xHistory[:, 2, :])))
        for i in range(xHistory.shape[2]):
            sc._offsets3d =  (np.ravel(xHistory[:, 0, i]),np.ravel(xHistory[:, 1, i]),np.ravel(xHistory[:, 2, i]))
            plt.gcf().suptitle('step #' + str(i) + '/' + str(xHistory.shape[2]-1))
            plt.pause(0.01)

    plt.ioff()
    plt.close()
