def three_point_bruteforce(dataset) :
    bestFit = float('-inf')
    bestI = 0
    bestJ = 1
    bestZ = 2
    for first in range(dataset.shape[0]) :
        for second in range(first,dataset.shape[0]) :
            if equality(dataset[first],dataset[second]) :
                continue
            for third in range(second,dataset.shape[0]) :

                if equality(dataset[first],dataset[third]) :
                    continue
                if equality(dataset[second],dataset[third]) :
                    continue
                fitness = calFitThird(dataset,first,second,third)
                if fitness > bestFit :
                    bestFit = fitness
                    bestI = first
                    bestJ = second  
                    bestZ = third
                if((first + second + third) % 300 == 0) :
                    print(bestFit)
    print('------------------------------------')
    print(bestFit)
    return bestFit,bestI,bestJ,bestZ


def calFitThird(dataset,first,second,third) :
    clusterMembers = np.array([None,None,None])
    for i,pattern in enumerate(clusterMembers) :
        clusterMembers[i] = np.array([]).astype(int)
    for i,data in enumerate(dataset):
        minDist = float('inf')
        minCentroidIndex = -1
        for j,centroid in enumerate([first,second,third]) :
            dist = ToolBox.euclideanDistance(dataset[i],dataset[centroid]) 
            if dist < minDist :
                minCentroidIndex = j
                minDist = dist
        clusterMembers[minCentroidIndex] = np.append(clusterMembers[minCentroidIndex],i)
    "Calculate each centroid point via mean of every cluster member"
    centroids = np.array([None,None,None])
    for i,cm in enumerate(clusterMembers):
        # calculating centroid for every cluster
        if len(cm) == 0 :       
            print(cm)
        centroids[i] = np.mean(dataset[cm],axis=0)
    
    s = [np.mean([ToolBox.euclideanDistance(pattern, member) \
      for member in dataset[clusterMembers[i]]]) for i,pattern in enumerate(centroids)]
    fitness = 1/np.mean([np.max([(s[i] + s[j])/ToolBox.euclideanDistance(pattern,pattern2) \
                    for j,pattern2 in enumerate(centroids) if i != j]) for i,pattern in enumerate(centroids)])
    return fitness

def two_point_bruteforce(dataset) : 
    bestFit = float('-inf')
    bestI = 0
    bestJ = 1
    for first in range(dataset.shape[0]) :
        for second in range(first,dataset.shape[0]) :
            if equality(dataset[first],dataset[second]) :
                continue
            fitness = calFit_2(dataset,first,second)
            if fitness > bestFit :
                bestFit = fitness
                bestI = first
                bestJ = second  
            if((first + second) % 100 == 0) :
                print(bestFit)
    return bestFit,bestI,bestJ
    
def calFit(dataset,first,second) :
    kmeans = KMeans(n_clusters=2,init=np.array([dataset[pattern] for pattern in [first,second]]),n_init=1).fit(dataset)
    clusterMembers = np.array([None,None])
    for i,pattern in enumerate(clusterMembers) :
        clusterMembers[i] = np.array([]).astype(int)
    for i,label in enumerate(kmeans.labels_) :
        clusterMembers[label] = np.append(clusterMembers[label],i)
    
    "Calculate each centroid point via mean of every cluster member"
    centroids = np.array([None,None])
    for i,cm in enumerate(clusterMembers):
        # calculating centroid for every cluster
        if len(cm) == 0 :       
            print(cm)
        centroids[i] = np.mean(dataset[cm],axis=0)
    
    s = [np.mean([ToolBox.euclideanDistance(pattern, member) \
      for member in dataset[clusterMembers[i]]]) for i,pattern in enumerate(centroids)]
    fitness = 1/np.mean([np.max([(s[i] + s[j])/ToolBox.euclideanDistance(pattern,pattern2) \
                    for j,pattern2 in enumerate(centroids) if i != j]) for i,pattern in enumerate(centroids)])
    return fitness

def calFit_2(dataset,first,second) :
#     kmeans = KMeans(n_clusters=2,init=np.array([dataset[pattern] for pattern in [first,second]]),n_init=1).fit(dataset)
    clusterMembers = np.array([None,None])
    for i,pattern in enumerate(clusterMembers) :
        clusterMembers[i] = np.array([]).astype(int)
    for i,data in enumerate(dataset):
        minDist = float('inf')
        minCentroidIndex = -1
        for j,centroid in enumerate([first,second]) :
            dist = ToolBox.euclideanDistance(dataset[i],dataset[centroid]) 
            if dist < minDist :
                minCentroidIndex = j
                minDist = dist
        clusterMembers[minCentroidIndex] = np.append(clusterMembers[minCentroidIndex],i)
#     print(clusterMembers)
    "Calculate each centroid point via mean of every cluster member"
    centroids = np.array([None,None])
    for i,cm in enumerate(clusterMembers):
        # calculating centroid for every cluster
        if len(cm) == 0 :       
            print(cm)
        centroids[i] = np.mean(dataset[cm],axis=0)
    
    s = [np.mean([ToolBox.euclideanDistance(pattern, member) \
      for member in dataset[clusterMembers[i]]]) for i,pattern in enumerate(centroids)]
    fitness = 1/np.mean([np.max([(s[i] + s[j])/ToolBox.euclideanDistance(pattern,pattern2) \
                    for j,pattern2 in enumerate(centroids) if i != j]) for i,pattern in enumerate(centroids)])
    return fitness

def equality(firstDp,secondDp) :
    if (len(firstDp) != len(secondDp)) :
        print("Error Length of array ")
        return None
    for i in range(len(firstDp)) :
        if firstDp[i] != secondDp[i] :
            return False
    return True
