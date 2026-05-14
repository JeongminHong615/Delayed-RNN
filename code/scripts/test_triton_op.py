"""Verify Triton fused buffer update matches the original update_buffer,
and benchmark it.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model.triton_ops import fused_buffer_update


def original_update_buffer(buffer, m_t1, delay_one_hot):
    """Same semantics as DRNN_jm.update_buffer but takes m_t1 as (B, H)."""
    # buffer: (B, MD, H)
    # m_t1: (B, H)
    # delay_one_hot: (B, MD, H)
    new_buffer = F.pad(buffer[:, 1:, :], (0, 0, 0, 1))  # shift left, pad zero at end
    values = m_t1.unsqueeze(1) * delay_one_hot  # (B, 1, H) * (B, MD, H) = (B, MD, H)
    return new_buffer + values


def test_forward():
    print("=" * 60)
    print("Forward correctness test")
    print("=" * 60)
    torch.manual_seed(0)
    for shape in [(2, 8, 4), (4, 32, 16), (128, 100, 64), (128, 200, 64)]:
        B, MD, H = shape
        buffer = torch.randn(B, MD, H, device="cuda")
        m_t1 = torch.randn(B, H, device="cuda")
        d_oh = torch.zeros(B, MD, H, device="cuda")
        # Make d_oh a one-hot over MD dim for each (B, H)
        delays = torch.randint(0, MD, (B, H), device="cuda")
        d_oh.scatter_(1, delays.unsqueeze(1), 1.0)

        ref = original_update_buffer(buffer, m_t1, d_oh)
        out = fused_buffer_update(buffer, m_t1, d_oh)

        diff = (ref - out).abs().max().item()
        print(f"shape (B={B}, MD={MD}, H={H}): max diff = {diff:.2e}", end="")
        if diff < 1e-5:
            print("  ✓")
        else:
            print("  ✗")


def test_backward():
    print()
    print("=" * 60)
    print("Backward correctness test (gradcheck via finite difference)")
    print("=" * 60)
    torch.manual_seed(0)
    B, MD, H = 8, 16, 8

    buffer = torch.randn(B, MD, H, device="cuda", requires_grad=True)
    m_t1 = torch.randn(B, H, device="cuda", requires_grad=True)
    d_oh = torch.zeros(B, MD, H, device="cuda")
    delays = torch.randint(0, MD, (B, H), device="cuda")
    d_oh.scatter_(1, delays.unsqueeze(1), 1.0)
    d_oh.requires_grad_(True)

    # Reference path
    buffer_r = buffer.detach().clone().requires_grad_(True)
    m_t1_r = m_t1.detach().clone().requires_grad_(True)
    d_oh_r = d_oh.detach().clone().requires_grad_(True)
    out_r = original_update_buffer(buffer_r, m_t1_r, d_oh_r)
    loss_r = out_r.sum()
    loss_r.backward()

    # Triton path
    out_t = fused_buffer_update(buffer, m_t1, d_oh)
    loss_t = out_t.sum()
    loss_t.backward()

    print(f"output diff:           {(out_r - out_t).abs().max().item():.2e}")
    print(f"grad_buffer diff:      {(buffer_r.grad - buffer.grad).abs().max().item():.2e}")
    print(f"grad_m_t1 diff:        {(m_t1_r.grad - m_t1.grad).abs().max().item():.2e}")
    print(f"grad_delay_one_hot diff: {(d_oh_r.grad - d_oh.grad).abs().max().item():.2e}")


def bench():
    print()
    print("=" * 60)
    print("Benchmark forward+backward")
    print("=" * 60)
    for B, MD, H in [(128, 100, 64), (128, 200, 64), (128, 130, 64)]:
        torch.manual_seed(0)
        buffer = torch.randn(B, MD, H, device="cuda", requires_grad=True)
        m_t1 = torch.randn(B, H, device="cuda", requires_grad=True)
        d_oh = torch.zeros(B, MD, H, device="cuda")
        delays = torch.randint(0, MD, (B, H), device="cuda")
        d_oh.scatter_(1, delays.unsqueeze(1), 1.0)
        d_oh.requires_grad_(True)

        def run_ref():
            buffer_local = buffer.detach().clone().requires_grad_(True)
            m_local = m_t1.detach().clone().requires_grad_(True)
            d_local = d_oh.detach().clone().requires_grad_(True)
            out = original_update_buffer(buffer_local, m_local, d_local)
            out.sum().backward()

        def run_tri():
            buffer_local = buffer.detach().clone().requires_grad_(True)
            m_local = m_t1.detach().clone().requires_grad_(True)
            d_local = d_oh.detach().clone().requires_grad_(True)
            out = fused_buffer_update(buffer_local, m_local, d_local)
            out.sum().backward()

        # Warmup
        for _ in range(5):
            run_ref(); run_tri()

        torch.cuda.synchronize(); t0 = time.time()
        N = 200
        for _ in range(N):
            run_ref()
        torch.cuda.synchronize(); t_ref = (time.time() - t0) / N * 1000

        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N):
            run_tri()
        torch.cuda.synchronize(); t_tri = (time.time() - t0) / N * 1000

        print(f"shape ({B},{MD},{H}): ref={t_ref:.3f}ms  triton={t_tri:.3f}ms  speedup={t_ref/t_tri:.2f}x")


if __name__ == "__main__":
    test_forward()
    test_backward()
    bench()
