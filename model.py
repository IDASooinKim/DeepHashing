# import main module
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

# import sub module
import inspect
import numpy as np

class Asig(nn.Module):
    def forward(self, x, alpha):
        return torch.sigmoid(alpha*x)

class Machine(nn.Module):

    def __init__(self, 
        in_size:int=768,
        hidden_size:int=768,
        hash_size:int=16,
        batch_size:int=32,
        task:str='gcn',
        k:int=1,
        device:str='cpu'
        ):

        """Machine learning model for graph-based deep hashing tasks.
        Args:
            in_size (int): Input size.
            hidden_size (int): Hidden layer size.
            hash_size (int): Hash size.
            batch_size (int): Batch size.
            task (str): Task type ('gcn', 'spline', 'fourier', 'cheby', 'lanczos').
            k (int): Number of iterations or basis functions.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """

        super(Machine, self).__init__()
        arguments = inspect.getfullargspec(self.__init__).args[1:]

        for arg in arguments:
            setattr(self, arg, locals()[arg])

        self.w1 = nn.Linear(in_size, hidden_size)
        self.w2 = nn.Linear(in_size, hash_size)
        self.w4 = nn.Linear(hidden_size, hidden_size)
        self.w5 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        self.sig = nn.Sigmoid()
        self.activation = nn.GELU()
        self.asig = Asig()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.cos = nn.CosineSimilarity
        
        # for fourier
        if task == 'gcn':
            self.w = nn.Linear(hidden_size, hidden_size)

        elif task == 'spline':
            self.w = nn.Linear(k, k)

        elif task == 'fourier':
            self.w = nn.Linear(k, k)

        elif task == 'cheby':
            self.w = nn.Parameter(data=torch.randn(k, hidden_size, hidden_size), requires_grad=True)
        
        elif task == 'lanczos':
            self.w = nn.Parameter(data=torch.randn(k, hidden_size, hidden_size), requires_grad=True)

        else:
            raise KeyError

    def adj_generator(self, B, device):

        # Compute pairwise cosine similarity
        B = torch.where(B>torch.mean(B),1,0).float()
        B_m = B-1
        A = 1 + (torch.matmul(B, B_m.T) + torch.matmul(B_m, B.T))/B.shape[1]
        A = torch.where(A>torch.mean(A),1,0).float()
        
        # diagnoal process
        D = torch.sum(A, dim=1)
        D = torch.diag(D)
        D = torch.diag(torch.diag(D).pow(-0.5))
        
        return A, D

    def fourier(self, L, k=1):
        eigenvalues, eigenvectors = torch.linalg.eigh(L) 

        # Select the k smallest eigenvalues and corresponding eigenvectors
        eigenvalues, indices = torch.topk(eigenvalues, k, largest=False)  # Get k smallest eigenvalues
        eigenvectors = eigenvectors[:, indices]

        return eigenvalues, eigenvectors

    def bspline_basis(self, K, x, degree):

        # Convert x to PyTorch tensor if it is a scalar
        if isinstance(x, int):
            x = torch.linspace(0, 1, x)
        elif isinstance(x, torch.Tensor):
            x = x.clone()  # Ensure we don't modify the original tensor

        # Evenly distributed knot vectors
        kv1 = x.min() * torch.ones(degree)
        kv2 = torch.linspace(x.min(), x.max(), K - degree + 1)
        kv3 = x.max() * torch.ones(degree)
        kv = torch.cat((kv1, kv2, kv3))

        # Define the recursive Cox - DeBoor function
        def cox_deboor(k, d):

            if d == 0:
                return ((x >= kv[k]) & (x < kv[k + 1])).int()

            denom1 = kv[k + d] - kv[k]
            term1 = torch.zeros_like(x)
            if denom1.abs() > 1e-10:  # Add an epsilon threshold
                term1 = ((x - kv[k]) / (denom1+1e-10)) * cox_deboor(k, d - 1)

            denom2 = kv[k + d + 1] - kv[k + 1]
            term2 = torch.zeros_like(x)
            if denom2.abs() > 1e-10:  # Add an epsilon threshold
                term2 = ((-(x - kv[k + d + 1]) / (denom2+1e-10)) * cox_deboor(k + 1, d - 1))

            return term1 + term2
        
        basis = torch.stack([cox_deboor(k, degree) for k in range(K)], dim=1)
        basis[-1, -1] = 1  # Ensure the last value is 1 as in the original code

        return basis

    def chebyshev_polynomials(self, L_hat, K):

        # Initialize the list of polynomials
        T_k = [torch.eye(L_hat.size(0)), L_hat.clone()]  # T0 and T1
        
        # Recursively compute T_k
        for k in range(2, K + 1):
            T_k_next = 2 * L_hat @ T_k[-1] - T_k[-2]
            T_k.append(T_k_next)
        
        return T_k

    def lanczos_algorithm(self, A, k, v0=None):

        n = A.shape[0]

        # if v0 is None:
        v0 = torch.randn(n)
        v0 = v0 / torch.norm(v0)
        
        alphas = []
        betas = []
        V = [v0]
        
        for i in range(k):
            w = torch.matmul(A, V[-1])
            alpha = torch.dot(V[-1], w)
            alphas.append(alpha.item())
            
            w = w - alpha * V[-1]
            if i > 0:
                w = w - betas[-1] * V[-2]
            
            beta = torch.norm(w)
            betas.append(beta.item())

            V.append(w / beta)
        
        T = torch.zeros(k, k)
        for i in range(k):
            T[i, i] = alphas[i]
            if i < k - 1:
                T[i, i+1] = betas[i+1]
                T[i+1, i] = betas[i+1]
                
        return T, torch.stack(V, dim=1)[:, :-1]

    def calc_sim(self, x):
        similarity_matrix = torch.matmul(x, x.T)/x.shape[0]
        return similarity_matrix

    def forward(self, x, alpha):

        # W1 process
        out1 = self.w1(x)
        out1 = self.activation(out1)
        out1_matrix = self.calc_sim(out1)

        # W2 process
        out2 = self.w2(x)
        out2 = self.sig(out2)

        # adj generation process
        if self.task == 'gcn':
            adj, deg = self.adj_generator(out1_matrix, self.device)
            out4 = torch.linalg.multi_dot([deg,adj,deg,out1])
            out4 = self.w(out4)
            out4 = self.layer_norm(out4)

        elif self.task == 'fourier':

            adj, deg = self.adj_generator(out1_matrix, self.device)
            L = torch.linalg.multi_dot([deg,adj,deg])
            lamb, U = self.fourier(L, k=self.k)
            out4 = self.w(U)
            out4 = torch.matmul(out4, U.T)
            out4 = torch.matmul(out4, out1)

        elif self.task == 'spline':

            adj, deg = self.adj_generator(out1_matrix, self.device)
            L = torch.linalg.multi_dot([deg,adj,deg])
            lamb, U = self.fourier(L, k=self.k)
            basis = self.bspline_basis(K=self.k, x=lamb, degree=3)
            out4 = torch.matmul(U, basis)
            out4 = self.w(out4)
            out4 = torch.matmul(out4, U.T)
            out4 = torch.matmul(out4, out1)
            
        elif self.task == 'cheby':

            adj, deg = self.adj_generator(out1_matrix, self.device)
            L = torch.linalg.multi_dot([deg,adj,deg])
            lamb, U = self.fourier(L, k=self.k)
            T_k = self.chebyshev_polynomials(L, K=self.k)
            out4 = T_k[0] @ out1  

            for k in range(1, self.k + 1):
                out4 += T_k[k] @ out1  

        elif self.task == 'lanczos':
            adj, deg = self.adj_generator(out1_matrix, self.device)
            L = torch.linalg.multi_dot([deg,adj,deg])
            T, V = self.lanczos_algorithm(L, self.k, out1)
            out4 = torch.linalg.multi_dot([V, T, V.T])
            out4 = torch.matmul(out4, out1)

        out4 = self.w4(out4)
        out4 = self.activation(out4)
    
        # decoder layers
        out4 = self.w5(out4) 

        return out4, out2, out1_matrix
