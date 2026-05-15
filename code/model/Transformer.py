"""Standard Transformer encoder baseline for delayed-copy task.

Uses:
  - input_proj: Linear(input_size, hidden_size)
  - learned positional embedding (up to a fixed max length)
  - n_layers of TransformerEncoderLayer (multi-head self-attention + FFN)
  - output_layer: Linear(hidden_size, output_size)

Non-causal attention: output positions attend to all positions, including
input data. Standard for sequence-to-sequence transduction with fixed-length
output.
"""
import torch
import torch.nn as nn
from utils.type import Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        id: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_id: str,
        device: str,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_seq_len: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_size, output_size)

        if dataset_id in ["delaysequence", "delaysequence_hard", "aec_synthetic", "seq_mnist"]:
            self.forward = self.SignalPeriod_forward
        else:
            raise NotImplementedError(
                f"Dataset {dataset_id} not supported for Transformer model."
            )

        self.to(self.device)

    def SignalPeriod_forward(
        self,
        x: Tensor,
        train: bool,
        logs: dict | None = None,
    ):
        bs, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bs, -1)
        x_proj = self.input_proj(x) + self.pos_emb(positions)
        out = self.encoder(x_proj)
        out = self.output_layer(out)
        return out, logs
