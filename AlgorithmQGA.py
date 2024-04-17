def quantumGeneticAlgorithm(dataset,popSize,pcc,pc,pm,pmm,preCriterion,catCriterion,iterNum,initPatternNum = None) :
    "main method, we will implement algorithm in it"
    bestFitArr = np.array([])
    bestFitCentArr = np.array([])
    bestChromosome = None
    bestSameIter = 0
    bestFitness = float('-inf')
    populations = np.full((iterNum),None)
    # randomly choosing initial pattern number in a chromosome in range(2,np.sqrt(N) + 1)
    if initPatternNum :
        initialPatternNum = initPatternNum
    else :
        initialPatternNum = random.randint(2,np.floor(np.sqrt(len(dataset)) + 1))
    PatternSize = int(np.ceil(np.log2(len(dataset))))
    populations[0] = Population(popSize,initialPatternNum,PatternSize)
    bestFit = populations[0].computeFitness(dataset)
    bestFitArr = np.append(bestFitArr,bestFit)
    bestFitCentArr = np.append(bestFitCentArr,len(populations[0].p[populations[0].bestChromosomeIndex].r))
    if bestFit > bestFitness :
        bestChromosome = copy.deepcopy(populations[0].p[populations[0].bestChromosomeIndex])
        bestFitness = bestFit
        bestSameIter = 0
    else :
        bestSameIter += 1
    for generation in range(1,iterNum) :   
        print('--------------------generation : {} ------------------'.format(generation))
        print('best fitness : {}'.format(bestFit))
        print('best chromosome cluster numbers : {}'.format(len(bestChromosome.r)))
        print('best chrom fit : {}'.format(bestFitness))
        print('best chrom cluster numbers : {}'.format(len(bestChromosome.r)))
        if bestSameIter < preCriterion :
            populations[generation - 1] = populations[generation - 1].selection()
#             populations[generation - 1] = populations[generation - 1].crossover(pc,'second',dataset)
            populations[generation - 1].mutate(pm)
            bestFit = populations[generation - 1].computeFitness(dataset)
            if bestFit > bestFitness :
                bestChromosome = copy.deepcopy(populations[generation - 1].p[populations[generation - 1].bestChromosomeIndex])
                bestFitness = bestFit
                bestSameIter = 0
            if bestSameIter < catCriterion :
                populations[generation - 1].rotate(bestChromosome)
            else :
                populations[generation - 1].catastrophe(bestChromosome)
                bestSameIter = 0
        else :
            populations[generation - 1] = populations[generation - 1].selection()
#             populations[generation - 1] = populations[generation - 1].crossover(pcc,'second',dataset)
            populations[generation - 1].mutate(pmm)
            bestFit = populations[generation - 1].computeFitness(dataset)
            if bestFit > bestFitness :
                bestChromosome = copy.deepcopy(populations[generation - 1].p[populations[generation - 1].bestChromosomeIndex])
                bestFitness = bestFit
                bestSameIter = 0
            if bestSameIter < catCriterion :
                populations[generation - 1].rotate(bestChromosome)
            else :
                populations[generation - 1].catastrophe(bestChromosome)
                bestSameIter = 0
        
        populations[generation] = populations[generation - 1]
        bestFit = populations[generation].computeFitness(dataset)
        bestFitCentArr = np.append(bestFitCentArr,len(populations[generation].p[populations[generation].bestChromosomeIndex].r))
        bestFitArr = np.append(bestFitArr,bestFit)
        if bestFit > bestFitness :
            bestChromosome = copy.deepcopy(populations[generation].p[populations[generation].bestChromosomeIndex])
            bestFitness = bestFit
            bestSameIter = 0
        else :
            bestSameIter += 1
    return bestFitArr,bestChromosome
