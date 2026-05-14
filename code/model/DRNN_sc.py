"""DRNN with x_t shortcut to delay_layer (variant of DRNN_jm).

Difference from DRNN_jm:
  - delay_layer receives both h_t AND raw x_t (concatenated).
  - Rationale: in the amended dataset the delay value is broadcast into
    channel k+1 of x_t at every step. Without the shortcut, this information
    has to pass through input_encoder -> emebed_t -> split/ReLU -> h_t
    (a 4-stage indirect path) before reaching delay_layer; that path mixes
    delay info with other content and halves it via ReLU.
  - With the shortcut, delay_layer can read x_t (and thus the broadcast
    delay value) directly. Expected to especially help dynamic-delay
    training, where the delay value varies per sample and must be re-read
    every batch.

Everything else is identical to DRNN_jm.
"""
import torch
import torch.nn as nn
from utils.type import Tensor
from jaxtyping import Float
import torch.nn.functional as F


class DRNN_sc(nn.Module):
    def __init__(
        self,
        id: str,
        max_delay: int,
        init_tau: float,
        min_tau: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_id: str,
        device: str,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_delay: int = max_delay
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.num_train: int = 0
        self.device: str = device
        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_tau: float = init_tau
        self.min_tau: float = min_tau
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.input_encoder = nn.Linear(self.input_size, 2 * self.hidden_size)
        self.m_encoder = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

        # delay_layer now reads BOTH x_t (raw) AND h_t.
        self.delay_layer = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.max_delay),
        )

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        if dataset_id in ["delaysequence"]:
            self.forward = self.SignalPeriod_forward
        else:
            raise NotImplementedError(f"Dataset {dataset_id} not supported.")

        self.to(self.device)

    @staticmethod
    def update_buffer(
        buffer: Float[Tensor, "b md h"],
        m_t1: Float[Tensor, "b h 1"],
        delay_steps: Float[Tensor, "b md h"],
        train: bool,
    ) -> Float[Tensor, "b md h"]:
        m_t1_t: Float[Tensor, "b 1 h"] = m_t1.transpose(1, 2)
        new_buffer: Float[Tensor, "b md h"] = F.pad(buffer[:, 1:, :], (0, 0, 0, 1))
        values: Float[Tensor, "b md h"] = (m_t1_t * delay_steps)
        return new_buffer + values

    def SignalPeriod_forward(
        self,
        x: Tensor,
        train: bool,
        logs: dict | None = None,
    ):
        bs, seq_len, _ = x.shape
        current_tau = self.init_tau

        if train:
            self.num_train += 1
            current_tau = max(self.min_tau, self.init_tau * (0.9999 ** self.num_train))
            if logs is not None:
                logs["model/tau"] = current_tau

        x_embed = self.input_encoder(x)
        outputs = x.new_zeros((bs, seq_len, self.output_size))
        buffer = x.new_zeros((bs, self.max_delay, self.hidden_size))
        delay_records = x.new_zeros((bs, seq_len))

        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t: Float[Tensor, "b h"] = self.tanh(buffer[:, 0, :])
            x_embed_t: Float[Tensor, "b 2h"] = x_embed[:, t, :]
            m_emebed_t: Float[Tensor, "b 2h"] = self.m_encoder(m_t)
            emebed_t: Float[Tensor, "b 2h"] = x_embed_t + m_emebed_t

            raw_memory: Float[Tensor, "b h"] = self.tanh(emebed_t[:, self.hidden_size:])
            h_t: Float[Tensor, "b h"] = self.relu(emebed_t[:, :self.hidden_size])
            m_t1: Float[Tensor, "b h 1"] = self.norm(raw_memory).unsqueeze(-1)

            # SHORTCUT: concat raw x_t and h_t for delay_layer input.
            delay_input = torch.cat([x_t, h_t], dim=-1)  # (b, input_size + hidden)
            shared_delay_logits = self.delay_layer(delay_input)
            delay_steps = shared_delay_logits.unsqueeze(1).expand(-1, self.hidden_size, -1)

            delay_one_hot: Float[Tensor, "b md h"] = F.gumbel_softmax(
                delay_steps, tau=current_tau, hard=True, dim=-1
            ).transpose(1, 2)

            chosen_delay = delay_one_hot.argmax(dim=1).float().mean(dim=-1)
            delay_records[:, t] = chosen_delay.float()

            buffer = self.update_buffer(buffer, m_t1, delay_one_hot, train)

            h_t_dropped = self.dropout_layer(h_t)
            out = self.output_layer(h_t_dropped)
            outputs[:, t, :] = out

        if logs is not None:
            logs["delay_records"] = delay_records
        return outputs, logs
