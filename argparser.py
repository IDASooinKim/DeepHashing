import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Deep Hashing Experiment Arguments")

    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--k', type=int, default=18, help='Hyperparameter k')
    parser.add_argument('--root_dir', type=str, default="/home/02_DeepHashing/dataset/Stanford_Car", help='Root directory of the dataset')
    parser.add_argument('--task', type=str, nargs='+', default=['lanczos', 'gcn', 'spline', 'cheby', 'fourier'], help='List of hashing tasks')
    parser.add_argument('--exp_hash_len', type=int, nargs='+', default=[16, 28, 32, 64, 128], help='List of hash code lengths to experiment with')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for computation')
    parser.add_argument('--num_cls', type=int, default=196, help='Number of classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()
