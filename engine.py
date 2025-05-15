import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from time import time
from utils.utils import *
from model import Machine

def run_experiment(k, num_epochs, task, exp_hash_len, train_loader, test_loader, query_loader, device):
    """
    Run the experiment for different hash lengths and tasks.
    
    Args:
        k (int): The order of the task.
        num_epochs (int): Number of epochs for training.
        task (list): List of tasks to run experiments on.
        exp_hash_len (list): List of hash lengths to experiment with.
    """
    # Experiment loop
    for hash_len in exp_hash_len:
        for key in task:
            machine = Machine(             
                        in_size=768,
                        hidden_size=768,
                        hash_size=hash_len,
                        task=key,
                        k=k
                        ).to(device)
            
            criterion = nn.MSELoss()  # Mean Squared Error Loss
            optimizer = optim.Adam(machine.parameters(), lr=0.0001)

            # Training steps
            print(f"Experment: {key} {hash_len}")
            best_mAP = 0
            
            for idx, epoch in enumerate(range(num_epochs)):
                start = time()
                machine.train()
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    outputs, hashes, adj = machine(x_batch, 1)

                    # Compute loss
                    loss1 = criterion(outputs, x_batch)
                    bin_hashes = 2*hashes-torch.ones_like(hashes)
                    loss3 = torch.sum(torch.mean((torch.mm(bin_hashes, bin_hashes.T) - hash_len*adj)**2))
                    loss = loss1 + 0.0001*loss3

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            # Inference steps
            with torch.no_grad():
                machine.eval()
                dataset_hcodes = []
                dataset_labels = []
                testset_hcodes = []
                testset_labels = []

                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    outputs, hashes, _ = machine(x_batch, 1)

                    dataset_hcodes.append(hashes.detach().cpu().numpy())
                    dataset_labels.append(y_batch.detach().cpu().numpy())
                
                dataset_hcodes = np.concatenate(dataset_hcodes, axis=0)
                dataset_hcodes = np.where(dataset_hcodes>=0.5, 1, 0)
                dataset_labels = np.concatenate(dataset_labels, axis=0)

                for x_batch, y_batch in query_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    outputs, hashes, _ = machine(x_batch, 1)

                    testset_hcodes.append(hashes.detach().cpu().numpy())
                    testset_labels.append(y_batch.detach().cpu().numpy())

                testset_hcodes = np.concatenate(testset_hcodes, axis=0)
                testset_hcodes = np.where(testset_hcodes>=0.5, 1, 0)
                testset_labels = np.concatenate(testset_labels, axis=0)

                dataset_hcodes = np.array((dataset_hcodes * 2) - 1, dtype=int)
                testset_hcodes = np.array((testset_hcodes * 2) - 1, dtype=int)
            
            mAP = CalcTopMap(
                rB=dataset_hcodes, 
                qB=testset_hcodes, 
                retrievalL=dataset_labels, 
                queryL=testset_labels, 
                topk=200
            )
            
            if best_mAP <= mAP:
                best_mAP = mAP

            df = pd.DataFrame({
                'map' : best_mAP,
                'model' : key,
                'k' : k,
                'hash_len' : hash_len
            }, index = [0])

            df.to_csv(f'./logs/{k}_{key}_{hash_len}.csv', index=False)
            print(f"mAP: {best_mAP}, model: {key}, order: {k}, hash_len: {hash_len}")