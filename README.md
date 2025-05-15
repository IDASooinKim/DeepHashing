<div align="center">

# Enhancing Visual Information Retrieval Systems via Graph Convolutional Networks Hashing with Transfer Learning and Spectral Filtering

This is a PyTorch implementation of a deep hashing algorithm integrated with a __mini-batch-based graph construction__ module. The provided experiments and dataset use [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). Please follow the instructions below to set up the experimental environment.

![poster](./images/arch.png)
<div align="left">

## Abstract

*To improve visual information retrieval systems such as image search, deep hashing techniques are used to generate compact representations of high-dimensional data. 
While convolutional neural networks (CNNs) have been the predominant approach for learning hash codes, their Euclidean space-based representations struggle to capture complex data structures effectively.
Conventional CNN-based hashing methods constrain data relationships to regular grid structures, which limits their ability to capture the complex structural information inherent in unstructured data or high-dimensional manifolds. 
Although GCN-based hashing approaches have been introduced to address these limitations, most rely on static graph structures, making it difficult to reflect the diversity of local data distributions.
We propose a graph convolutional autoencoder that combines transfer learning-based visual embeddings with a range of spectral filtering strategies. 
For each input batch, the model dynamically constructs local subgraphs and exploits the eigenvectors of the graph Laplacian to extract both global and local structural features from the graph signals. 
Through this process, the graph convolutional layers can effectively model the local topology of unstructured data while simultaneously improving learning efficiency and representation capacity through parameter sharing and transfer learning.
Experiments conduct on the STL-10, Stanford Cars, and Tiny ImageNet datasets demonstrate that the proposed model achieves competitive performance not only against traditional CNN-based hashing methods but also against GCN-based methods using static graphs. 
In particular, the application of diverse spectral filters quantitatively reveals differences in graph representation capacity.
This work empirically demonstrates that spectral-domain filtering operations contribute to preserving structural information and enhancing representation power in graph-based deep hashing. 
Moreover, it highlights the effectiveness of combining dynamic graph construction with transfer learning for representing high-dimensional image data.*
