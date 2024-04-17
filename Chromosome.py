class Chromosome :
    def __init__(self,cSize,iSize) :
        self.size = cSize
        self.iSize = iSize
        self.fitness = float('-inf')
        self.r = np.array([])
        if (cSize == 0) :
            print(cSize)
            print('hi babe')
        for _ in range(cSize) :
            self.r = np.append(self.r,Pattern(iSize))
    
    def __str__(self) :
        return str([p.real for p in self.r])
    
    def __len__(self) :
        return self.iSize * self.size
    
    def toReal(self,dpSize,dataset) :
        "transforming each pattern to real number"
        for p in self.r :
            p.toReal(dpSize)
        isEqualReal = True
        while isEqualReal :
            isEqualReal = False
            #check if there is two pattern with same real number
            for p in self.r :
                for p2 in self.r :
                    if p != p2 :
                        while np.min([i == j for i,j in zip(dataset[p.real],dataset[p2.real])]) :
                            isEqualReal = True
                            p.toReal(dpSize)

    def rotate(self,b,isGreater) :
        for i,pattern in enumerate(self.r) :
            pattern.rotate(b.r[i],isGreater)
            
    def mutate(self) :
        "generating a random number and change a and b in position of rnd in chromosome"
        rnd = np.random.randint(0,self.__len__())
        pos = self.patternPos(rnd)
        self.r[pos[0]].mutate(pos[1])
        
    def computeFitness(self,dataset) :
        self.toReal(len(dataset),dataset)
#         self._kMeansClustering(dataset)
        self._alocateCluster(dataset)
        self._calculateCentroid(dataset)
        s = [np.mean([ToolBox.euclideanDistance(pattern.centroid, member) \
                      for member in dataset[pattern.clusterMembers]]) for pattern in self.r]
        self.fitness = 1/np.mean([np.max([(s[i] + s[j])/ToolBox.euclideanDistance(pattern.centroid,pattern2.centroid) \
                                for j,pattern2 in enumerate(self.r) if i != j]) for i,pattern in enumerate(self.r)])
        return self.fitness
       
    def _kMeansClustering(self,dataset) :
                    
        kmeans = KMeans(n_clusters=self.size,init=np.array([dataset[pattern.real] for pattern in self.r]),n_init=1).fit(dataset)
#         kmeans = KMeans(n_clusters=self.size).fit(dataset)

        "Alocating data points to each cluster via their euclidean distance"
        for i,pattern in enumerate(self.r) :
            self.r[i].clusterMembers = np.array([]).astype(int)
        
        for i,label in enumerate(kmeans.labels_) :
            self.r[label].clusterMembers = np.append(self.r[label].clusterMembers,i)
        
        "Calculate each centroid point via mean of every cluster member"      
        self._calculateCentroid(dataset)
#         for i,pattern in enumerate(self.r):
#             # calculating centroid for every cluster
#             self.r[i].centroid = kmeans.cluster_centers_[i]
                    
    def _alocateCluster(self,dataset) :
        "Alocating data points to each cluster via their euclidean distance"
        for i,centroid in enumerate(self.r) :
            self.r[i].clusterMembers = np.array([]).astype(int)
        
        for i,data in enumerate(dataset):
            minDist = float('inf')
            minCentroidIndex = -1
            for j,centroid in enumerate(self.r) :
                dist = ToolBox.euclideanDistance(dataset[i],dataset[centroid.real]) 
                if dist < minDist :
                    minCentroidIndex = j
                    minDist = dist
            self.r[minCentroidIndex].clusterMembers = np.append(self.r[minCentroidIndex].clusterMembers,i)
            
    def _calculateCentroid(self,dataset) :
        "Calculate each centroid point via mean of every cluster member"
        for pattern in self.r:
            # calculating centroid for every cluster
            if len(pattern.clusterMembers) == 0 :       
                print(pattern.clusterMembers)
            pattern.centroid = np.mean(dataset[pattern.clusterMembers],axis=0)
                           
    def patternPos(self,qbitPos) :
        """calculating the position of the qbit in pattern
        return (patternPos,qbitInPatternPos)"""

        if qbitPos >= self.__len__() :
            print('warning ' + str(qbitPos))
            print(int(np.floor(qbitPos/self.iSize)),int(qbitPos % self.iSize))
            
        return (int(np.floor(qbitPos/self.iSize)),int(qbitPos % self.iSize))
