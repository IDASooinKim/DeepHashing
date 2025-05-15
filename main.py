# import main modules
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import StanfordCars, ImageFolder
# import sub modules
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

# import usr defined modules
from utils.data_loader import *
from engine import run_experiment
from time import time

# Usage example
num_epochs = 20
batch_size = 128
k = 18
root_dir = "/home/02_DeepHashing/dataset/Stanford_Car"
task = ['lanczos', 'gcn', 'spline', 'cheby', 'fourier']
exp_hash_len = [16, 28, 32, 64, 128]
device = 'cpu'
num_cls = 196
seed = 42

######## Data loading ########
# all_data = NpyFolderDataset(
#     root_dir=root_dir, 
#     num_cls=num_cls
#     )

train_loader, test_loader, query_loader = get_data_loader(    
    root_dir=root_dir, 
    batch_size=batch_size, 
    seed=seed, 
    num_cls=num_cls
    )

run_experiment(
    k, 
    num_epochs, 
    task, 
    exp_hash_len, 
    train_loader, 
    test_loader, 
    query_loader,
    device
    )

########################### experiment 1: search proper k ###########################
