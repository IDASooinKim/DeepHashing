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

## Installation

To train the model, you need to set up the experimental environment. Use a virtual environment such as Anaconda to install the packages listed in the provided requirements.txt file. While various versions of Python modules may be used, the versions specified in the text file reflect the environment used in the experiments.

### 1. Clone the repository

```{shell}
git clone https://github.com/IDASooinKim/DeepHashing.git
```

### 2. Creating conda envs

```{shell}
conda create -n deephashing python=3.8
conda activate deephashing
```

### 3. Install requirements 

```{shell}
pip install -r requirements.txt
```

## Training Data preparation

Please download the image dataset for the experiment. 
The downloaded images must be embedded into 1-dimensional vectors of size 784 using the [ViT B/16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) model. 
For faster experimentation, you can download the pre-embedded [Stanford_Cars dataset](https://drive.google.com/file/d/1s39IUmYMnvvwMu1eotckh3HF6Mr1QvUt/view?usp=drive_link).

Each folder in the downloaded dataset represents a class, and each folder contains approximately 50 embeddings of the same class. If you wish to train with custom data, please follow the folder directory structure below.

```
project/
├── data/
│   ├── 0
|   |───── 0.npy
|   |───── 1.npy
│   └── 1
```

## Model Train

Model training can be easily run using the command __python main.py__.
If you wish to adjust the batch size, number of epochs, or other parameters, please use the following command:

```{shell}
python main.py --num_epochs 200 --batch_size 64 --num_cls 10 
```

For detailed arguments, please refer to the arguments.py file.

## Evaluation

The model's mAP results are printed to the CLI and saved as a .csv file in the "logs" folder of the project directory after each experiment is completed.
The mAP comparison table presented in the paper is as follows.

### STL-10
| Model   | 16-bit | 28-bit | 32-bit | 64-bit | 128-bit |
| ------- | ------ | ------ | ------ | ------ | ------- |
| Poly    | 0.645  | 0.801  | 0.873  | 0.882  | 0.881   |
| Lanczos | 0.860  | 0.870  | 0.873  | 0.878  | 0.882   |
| Cheby   | 0.861  | 0.870  | 0.872  | 0.872  | 0.881   |
| GCN     | 0.861  | 0.869  | 0.872  | 0.882  | 0.883   |

### Stanford Cars
| Model   | 16-bit | 28-bit | 32-bit | 64-bit | 128-bit |
| ------- | ------ | ------ | ------ | ------ | ------- |
| Poly    | 0.835  | 0.851  | 0.913  | 0.932  | 0.933   |
| Lanczos | 0.900  | 0.918  | 0.918  | 0.927  | 0.934   |
| Cheby   | 0.903  | 0.920  | 0.919  | 0.927  | 0.936   |
| GCN     | 0.911  | 0.918  | 0.920  | 0.932  | 0.952   |

### Tiny ImageNet
| Model   | 16-bit | 28-bit | 32-bit | 64-bit | 128-bit |
| ------- | ------ | ------ | ------ | ------ | ------- |
| Poly    | 0.897  | 0.910  | 0.945  | 0.968  | 0.979   |
| Lanczos | 0.914  | 0.935  | 0.959  | 0.971  | 0.982   |
| Cheby   | 0.929  | 0.938  | 0.955  | 0.970  | 0.983   |
| GCN     | 0.934  | 0.940  | 0.967  | 0.975  | 0.989   |
