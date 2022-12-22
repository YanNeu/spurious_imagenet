import torch
import numpy as np
import random

GLOBAL_SEED = 0

def set_rand_seed(seed=GLOBAL_SEED):
    """
    Sets the random seed for torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def print_used_seed():
    """
    Prints currently used seeds for torch, numpy, and random.
    """
    print(f'Pytorch seed: {torch.initial_seed()}')
    print(f'Cuda seed: {torch.cuda.initial_seed()}')
    print(f'NumPy seed: {np.random.get_state()[1][0]}')
