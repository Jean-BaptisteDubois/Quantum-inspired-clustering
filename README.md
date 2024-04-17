# Quantum-inspired-clustering
This repository focuses on two main quantum-inspired clustering methods ; namely the original "quantum clustering" introduced by Horn in 2001 and a quantum inspired genetic algorithm. 
This work was intended to be done in a scholar framework, a detailed report has so been written (containing exhaustive descriptions of the two methods and comparing the numerical results obtained), hence the sparse informations provided on this page. 

Two main python implementations are provided. 

The first one focuses on the unsupervised learning algorithm developed by Horn and Gottlieb, which tries to build a potential function from a given data set in order to constiute clusters within the data. The main idea is to associate the minima of this potential function to the center of the clusters (according to quantum mechanics based intuitions), then realise a gradient descent to make our data points converge towards these centers. 

The original (and seminal paper) can be found as : Horn, D., & Gottlieb, A. (2001). Quantum clustering. arXiv preprint physics/0107063.
