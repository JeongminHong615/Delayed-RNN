import torch 
import torch.nn as nn 
from utils.type import Tensor

class RNN(nn.Module):
    def __init__(
        self,
        id: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_id: str,
        device: str,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.device: str = device
        self.num_layers: int = num_layers
        self.dropout: float = dropout
        
        self.model = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        if dataset_id in ["delaysequence", "delaysequence_hard"]:
            self.forward = self.SignalPeriod_forward
        else:
            raise NotImplementedError(f"Dataset {dataset_id} not supported for RNN model.")
        
        self.to(self.device)
        
    def SignalPeriod_forward(
            self,
            x: Tensor,
            train: bool,
            logs: dict[str, Tensor]
    ):
        out, _ = self.model(x)
        out = self.output_layer(out)
        return out, logs
        