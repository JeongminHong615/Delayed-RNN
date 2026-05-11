import torch
import random
import numpy as np
from torch.utils.flop_counter import FlopCounterMode

def set_seed(seed: int)-> None:
    """ Set random seed for reproducibility across various libraries and frameworks.
    Args:
        seed (int): The random seed to set for reproducibility.
    """
    random.seed(seed) # Python 
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # Multi-GPU
    
    torch.backends.cudnn.deterministic = True # CUDNN's deterministic mode
    torch.backends.cudnn.benchmark = False # CUDNN's benchmark mode



def calculate_flops(
    model: torch.nn.Module,
    input_size: int,
    seq_len: int,
    device: str,
):
    input = torch.ones(1, seq_len, input_size).to(device) # 1로 채워진 가짜 데이터 만들기
    with FlopCounterMode(display=True) as fcm: # 연산량 측정
        model(input, train=False, logs=None)
        total_flops = fcm.get_total_flops()
    
    return total_flops / 1e6 # Return FLOPs in millions