import os
import numpy as np
import torch 

from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.model_selection import train_test_split
from glob import glob


class NpyFolderDataset(Dataset):
    def __init__(
        self, 
        root_dir:str, 
        num_cls:int=10
        ):
        r"""
            Dataset for loading .npy files from a folder structure.

            Args:
                root_dir (str): Directory with all the .npy files.
                transform (callable, optional): Optional transform to be applied on a sample.
                num_cls (int): Number of classes. Default is 10.
        """
        super(NpyFolderDataset, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.targets = []
        self.classes = []
        
        # Mapping class names to indices
        if num_cls == 10:
            for class_num in range(1, num_cls+1):
                class_dir = os.path.join(root_dir, str(class_num))
                if os.path.isdir(class_dir):
                    for file_name in os.listdir(class_dir):
                        if file_name.endswith('.npy'):
                            file_path = os.path.join(class_dir, file_name)
                            self.data.append(file_path)
                            self.targets.append(class_num-1)
        else:
            for class_num in range(0, num_cls):
                class_dir = os.path.join(root_dir, str(class_num))
                if os.path.isdir(class_dir):
                    for file_name in os.listdir(class_dir):
                        if file_name.endswith('.npy'):
                            file_path = os.path.join(class_dir, file_name)
                            self.data.append(file_path)
                            self.targets.append(class_num-1)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # Load the .npy file
        file_path = self.data[idx]
        sample = np.load(file_path)
        
        # Convert to tensor
        sample = torch.tensor(sample, dtype=torch.float)

        label = self.targets[idx]
        
        return sample.squeeze(), label


def get_data_loader(
    root_dir:str, 
    batch_size:int, 
    seed:int=42, 
    num_cls:int=196
    ):

    """
    Function to get data loaders for training, testing, and querying.
    Args:
        root_dir (str): Directory with all the .npy files.
        batch_size (int): Batch size for the DataLoader.
        seed (int): Random seed for splitting the dataset.
        num_cls (int): Number of classes. Default is 196.
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
        query_loader (DataLoader): DataLoader for the query set.  
    """

    dataset_size = len(glob(f'{root_dir}/**/*'))
    indices = list(range(dataset_size))

    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=seed
        )
    db_indices, query_indices = train_test_split(
        test_indices, 
        test_size=0.5, 
        random_state=seed
        )

    dataset = NpyFolderDataset(
        root_dir, 
        num_cls=num_cls
        )

    train_dataset = Subset(
        dataset, 
        train_indices
        )
    test_dataset = Subset(
        dataset, 
        db_indices
        )
    query_dataset = Subset(
        dataset, 
        query_indices
        )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
        )
    query_loader = DataLoader(
        query_dataset, 
        batch_size=batch_size, 
        shuffle=False
        )

    return train_loader, test_loader, query_loader
