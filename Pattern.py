class Pattern :
    def __init__(self,size) :
        self.size = size       
        self.clusterMembers = None
        self.centroid = None
        self.id = np.array([])
        for _ in range(size) :
            self.id = np.append(self.id,Qbit())
        self.real = None
           
    def __str__(self) :
        return '({})'.format(self.real)
    
    def toReal(self,maxValue) :
        "transform pattern to real number which is centroid position"
        self.real = 0
        for i,qbit in enumerate(self.id) :
            self.real += qbit.toBit()*np.power(2,self.size - i - 1)
        self.real = ToolBox.translate(
                self.real,0,
                np.power(2,len(self.id))-1,
                0,maxValue) # fit generated number to length of the dataset
        return self.real
    
    def rotate(self,b,isGreater) :
        "rotate each qbit"
        for i,qbit in enumerate(self.id) :
            qbit.rotate(b.id[i],isGreater)
            
    def mutate(self,pos) :
        "will mutate the qbit in position {pos} in pattern"
        self.id[pos].mutate()
