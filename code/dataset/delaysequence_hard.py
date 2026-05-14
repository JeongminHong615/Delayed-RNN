"""DelaySequence (hard variant) — pre-amend style, no delay broadcast.

This restores the dataset format from BEFORE the `amend delaysequence` commit:
  - channel k    : End token (binary 1 at position seq_len-1)
  - channel k+1  : Hint token (binary 1 at position start_output-1)
  - NO broadcast of the delay value anywhere

The model can detect "input ended" from End token, and "output starts next"
from Hint token. But the delay value itself is NOT given as an input feature.
The model must infer delay from positional timing between End and Hint, which
appears only AFTER all input data has been processed.

Architectural test: LD-RNN must decide write-time delay BEFORE seeing Hint,
so it should structurally fail without the broadcast. LSTM, with general
cell-state memory, may still partially learn from End/Hint markers and
sequence statistics.

Train/eval/loss reuse delaysequence.py implementations.
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.delaysequence import (
    DelaySequenceDataset,
    compute_loss,
    logging,
    remove_list_log,
)


class DelaySequenceHardDataset(DelaySequenceDataset):
    def __init__(
        self,
        id,
        size,
        min_len=5,
        max_len=100,
        k=10,
        delay=10,
        min_delay=None,
        max_delay=None,
        train=True,
    ):
        # Use parent constructor; same dimensions / params apply.
        super().__init__(
            id=id,
            size=size,
            min_len=min_len,
            max_len=max_len,
            k=k,
            delay=delay,
            min_delay=min_delay,
            max_delay=max_delay,
            train=train,
        )

    def __getitem__(self, idx):
        seq_len = random.randint(self.min_len, self.max_len)
        delay = random.randint(self.min_delay, self.max_delay)

        seq = [random.randint(0, self.k - 1) for _ in range(seq_len)]
        seq_tensor = torch.tensor(seq, dtype=torch.long)

        input_seq = torch.zeros(self.total_len, self.input_size)
        input_seq[:seq_len, :self.k] = F.one_hot(
            seq_tensor, num_classes=self.k
        ).float()

        # End token at last input position — channel k.
        input_seq[seq_len - 1, self.k] = 1.0

        # Hint token at start_output - 1 — channel k+1 (binary, not broadcast).
        start_output = seq_len + delay
        input_seq[start_output - 1, self.k + 1] = 1.0

        target_seq = torch.full((self.total_len,), -1, dtype=torch.long)
        target_seq[start_output:start_output + seq_len] = seq_tensor

        return input_seq, target_seq
