class Population :
    def __init__(self,pSize,cSize,iSize) :
        self.size = pSize
        self.cSize = cSize
        self.iSize = iSize
        self.bestChromosomeIndex = None
        self.p = np.array([])
        for _ in range(pSize) :
            self.p = np.append(self.p,Chromosome(cSize,iSize))

    def toReal(self) :
        for c in self.p :
            c.toReal()

    def rotate(self,bestChoromosome) :
        
        for chromosome in self.p :
            # checking if the length of the chromosomes are the same as th
            if len(chromosome) > len(bestChoromosome) :
                chrom = Chromosome(int(np.ceil(len(bestChoromosome)/self.iSize)),self.iSize)
                for i in range(len(chrom)) :
                    chromPatternPos = chrom.patternPos(i)
                    chrom.r[chromPatternPos[0]].id[chromPatternPos[1]] = chromosome.r[chromPatternPos[0]].id[chromPatternPos[1]]  
                chromosome = chrom
            elif len(chromosome) < len(bestChoromosome) :
                chrom = Chromosome(int(np.ceil(len(bestChoromosome)/self.iSize)),self.iSize)
                for i in range(len(chromosome)) :
                    chromPatternPos = chrom.patternPos(i)
                    chrom.r[chromPatternPos[0]].id[chromPatternPos[1]] = chromosome.r[chromPatternPos[0]].id[chromPatternPos[1]]  
                chromosome = chrom
            chromosome.rotate(bestChoromosome,bestChoromosome.fitness > chromosome.fitness)
        
    def mutate(self,prob) :
        for chromosome in self.p :
            rnd = random.random()
            if rnd < prob :
                chromosome.mutate()

    def computeFitness(self,dataset) :
        maxFit = float('-inf')
        for i,chromosome in enumerate(self.p) :
            chromosome.computeFitness(dataset)
            if chromosome.fitness > maxFit :
                maxFit = chromosome.fitness
                self.bestChromosomeIndex = i
        return maxFit

    def eliteSelection(self,population) :
        maxFit = np.max([ch.fitness for ch in population.p])
        if self.p[self.bestChromosomeIndex].fitness > maxFit :
            for ch in sorted(self.p, key=lambda x: x.fitness) :
                if ch.fitness > maxFit :
                    population.p[np.random.randint(0,self.size)] = copy.deepcopy(ch)
        return population

    def selection(self) :    
        population = Population(self.size,self.cSize,self.iSize)
        # Roulette selection
        for i in range(self.size) :
            population.p[i] = copy.deepcopy(self.roulette())
        # Elite selection
        maxFit = np.max([ch.fitness for ch in population.p])
        if self.p[self.bestChromosomeIndex].fitness > maxFit :
            for ch in sorted(self.p, key=lambda x: x.fitness) :
                if ch.fitness > maxFit :
                    population.p[np.random.randint(0,self.size)] = copy.deepcopy(ch)
        return population 
    
    def roulette(self) :
        sumFit = np.sum([ch.fitness for ch in self.p])
        pick = random.uniform(0, sumFit)
        current = 0
        for chromosome in self.p:
            current += chromosome.fitness
            if current > pick:
                return chromosome

    def catastrophe(self,bestChromosome) :
        self.__init__(self.size,self.cSize,self.iSize)
        self.p[0] = copy.deepcopy(bestChromosome)
        
    def crossover(self,prob,method='first',dataset=None) :
        population = Population(self.size,self.cSize,self.iSize)
        for i in range(int(self.size/2)) :
            self._mating(prob,population,i,method)
        if method == 'first' :
            population.computeFitness(dataset)
            return self.eliteSelection(population)
        else :
            return population
    
    def _mating(self,prob,population,j,method='first') :
        firstPoint = 0
        secondPoint = 0
        isDiffrentParent = False
        if method == 'first' :
            parent1 = copy.deepcopy(self.roulette())
            parent2 = copy.deepcopy(self.roulette())
        else :
            parent1 = copy.deepcopy(self.p[np.random.randint(0,self.size)])
            parent2 = copy.deepcopy(self.p[np.random.randint(0,self.size)])
        # finding the standard points for crossover
        if random.random() <= prob:
            isStandardPoint = False
            while (not isStandardPoint) :
                firstPoint = np.random.randint(0,len(parent1))
                secondPoint = np.random.randint(0,len(parent2))
                firstChildLen = (firstPoint + len(parent2) - secondPoint)
                secondChildLen = (len(parent1) - firstPoint + secondPoint)
                isStandardPoint = (firstChildLen % self.iSize == 0) and (firstChildLen/self.iSize > 1) and \
                    (secondChildLen % self.iSize == 0) and (secondChildLen/self.iSize > 1)
        # 2 point crossover
        firstChildLen = firstPoint + len(parent2) - secondPoint
        secondChildLen = len(parent1) - firstPoint + secondPoint
        child1 = Chromosome(int(np.ceil(firstChildLen/self.iSize)),self.iSize)
        child2 = Chromosome(int(np.ceil(secondChildLen/self.iSize)),self.iSize)
        for i in range(firstChildLen) :
            childPatternPos = child1.patternPos(i)
            if i < firstPoint :
                parentPatternPos = parent1.patternPos(i)
                child1.r[childPatternPos[0]].id[childPatternPos[1]] = parent1.r[parentPatternPos[0]].id[parentPatternPos[1]]
            else :
                parentPatternPos = parent2.patternPos(secondPoint + (i - firstPoint))
                child1.r[childPatternPos[0]].id[childPatternPos[1]] = parent2.r[parentPatternPos[0]].id[parentPatternPos[1]]

        for i in range(secondChildLen) :
            childPatternPos = child2.patternPos(i)
            if i < secondPoint :
                parentPatternPos = parent2.patternPos(i)
                child2.r[childPatternPos[0]].id[childPatternPos[1]] = parent2.r[parentPatternPos[0]].id[parentPatternPos[1]]
            else :
                parentPatternPos = parent1.patternPos(firstPoint + (i - secondPoint))
                child2.r[childPatternPos[0]].id[childPatternPos[1]] = parent1.r[parentPatternPos[0]].id[parentPatternPos[1]]
        population.p[2*j] = copy.deepcopy(child2)                
        population.p[2*j+1] = copy.deepcopy(child1)     
