"""DRNN v3 — continuous interpolation kernel (no Gumbel-Softmax).

Instead of selecting one buffer slot via discrete Gumbel-Softmax, each neuron
has a *continuous* delay value d_j in [0, max_delay-1] and the memory is
written to neighboring slots with weights from a triangular kernel:

    weight[p, j] = relu(1 - |p - d_j| / sigma)

Then weights are normalized so each neuron's total written mass equals 1.
Gradients flow directly into d_j without Gumbel noise — analogous to
Hammouamri et al. (DCLS) in SNNs, adapted to continuous-value RNN buffers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from utils.type import Tensor


class DRNN_v3(nn.Module):
    def __init__(
        self,
        id: str,
        max_delay: int,
        init_sigma: float,
        min_sigma: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_id: str,
        device: str,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_delay = max_delay
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_train = 0
        self.device = device
        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_sigma = init_sigma
        self.min_sigma = min_sigma
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.input_encoder = nn.Linear(self.input_size, 2 * self.hidden_size)
        self.m_encoder = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

        # Per-neuron continuous delay: shared input-dependent scalar per neuron
        # plus a learnable per-neuron bias. Final delay ∈ [0, max_delay-1] via
        # sigmoid scaling.
        self.delay_shared = nn.Linear(self.hidden_size, self.hidden_size)
        self.delay_neuron_bias = nn.Parameter(torch.zeros(self.hidden_size))

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        # Cached buffer-slot indices [0, 1, ..., max_delay-1] as a float tensor.
        self.register_buffer(
            "_positions",
            torch.arange(self.max_delay, dtype=torch.float32),
        )

        if dataset_id in ["delaysequence", "delaysequence_v2"]:
            self.forward = self.SignalPeriod_forward
        else:
            raise NotImplementedError(f"Dataset {dataset_id} not supported.")

        self.to(self.device)

    def update_buffer(
        self,
        buffer: Float[Tensor, "b md h"],
        m_t1: Float[Tensor, "b h"],
        delay_value: Float[Tensor, "b h"],
        current_sigma: float,
    ) -> Float[Tensor, "b md h"]:
        # positions: (md,) ; delay_value: (b, h)
        # diff[b, p, h] = positions[p] - delay_value[b, h]
        diff = self._positions[None, :, None] - delay_value[:, None, :]
        # Triangular kernel: relu(1 - |diff|/sigma)
        weights = F.relu(1.0 - torch.abs(diff) / current_sigma)
        # Normalize so each (b, h)'s weights sum to 1 (preserve total mass)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # values[b, p, h] = m_t1[b, h] * weights[b, p, h]
        values = m_t1[:, None, :] * weights

        # Shift buffer left (oldest slot = 0 falls off; pad zero at the end)
        new_buffer = F.pad(buffer[:, 1:, :], (0, 0, 0, 1))
        return new_buffer + values

    def SignalPeriod_forward(
        self,
        x: Tensor,
        train: bool,
        logs: dict | None = None,
    ):
        bs, seq_len, _ = x.shape
        current_sigma = self.init_sigma

        if train:
            self.num_train += 1
            current_sigma = max(
                self.min_sigma, self.init_sigma * (0.9999 ** self.num_train)
            )
            if logs is not None:
                logs["model/sigma"] = current_sigma

        x_embed = self.input_encoder(x)
        outputs = x.new_zeros((bs, seq_len, self.output_size))
        buffer = x.new_zeros((bs, self.max_delay, self.hidden_size))
        delay_records = x.new_zeros((bs, seq_len))

        for t in range(seq_len):
            m_t: Float[Tensor, "b h"] = self.tanh(buffer[:, 0, :])
            x_embed_t = x_embed[:, t, :]
            m_emebed_t = self.m_encoder(m_t)
            emebed_t = x_embed_t + m_emebed_t

            raw_memory = self.tanh(emebed_t[:, self.hidden_size:])
            h_t = self.relu(emebed_t[:, :self.hidden_size])
            m_t1 = self.norm(raw_memory)  # (b, h)

            # Per-neuron continuous delay value ∈ [0, max_delay-1].
            raw = self.delay_shared(h_t) + self.delay_neuron_bias[None, :]
            delay_value = torch.sigmoid(raw) * (self.max_delay - 1)

            # Diagnostic: mean delay across neurons (for monitoring).
            delay_records[:, t] = delay_value.mean(dim=-1)

            buffer = self.update_buffer(buffer, m_t1, delay_value, current_sigma)

            h_t_dropped = self.dropout_layer(h_t)
            outputs[:, t, :] = self.output_layer(h_t_dropped)

        if logs is not None:
            logs["delay_records"] = delay_records

        return outputs, logs
