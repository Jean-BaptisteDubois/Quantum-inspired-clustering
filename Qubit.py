class Qbit :
    def __init__(self) :
        self.a = random.random()
        self.b = np.sqrt(1 - self.a**2)
        self.bit = None
    
    def __str__(self) :
        return '({}, {})'.format(self.a,self.b)

    def mutate(self) :
        self.a,self.b = self.b,self.a
        
    def toBit(self) :
        "transform qbit to zero or one"
        if random.random() < self.a**2 :
            self.bit = 0
        else :
            self.bit = 1
        return self.bit     
            
    def rotate(self,bi,isGreater) :
        dt = 0
        sign = 0
        ri = self.bit
        positive = self.a * self.b > 0
        aZero = not self.a
        bZero = not self.b
        # initializing angle and sign of rotation 
        if(isGreater) :
            if not ri and bi :
                dt = np.pi * .05
                if aZero :
                    sign = 1
                elif bZero :
                    sign = 0
                elif positive :
                    sign = -1
                else :
                    sign = 1
            elif ri and not bi :
                dt = np.pi * .025
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1
            elif ri and bi :
                dt = np.pi * .025
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1
        else :
            if ri and not bi :
                dt = np.pi * .01
                if aZero :
                    sign = 1
                elif bZero :
                    sign = 0
                elif positive :
                    sign = -1
                else :
                    sign = 1
            elif ri and bi :
                dt = np.pi * .005
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1
        
        t = sign * dt
        self.a,self.b = np.dot(
            np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]),np.array([self.a,self.b])
        )
            
