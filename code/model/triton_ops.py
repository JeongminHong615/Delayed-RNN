"""Triton-fused buffer update for LD-RNN.

Original op (update_buffer):
    new_buffer[b, j, :] = (buffer[b, j+1, :] if j+1 < MD else 0)
                          + m_t1[b, :] * delay_one_hot[b, j, :]

This fuses:
  1. F.pad shift           -> O(B*MD*H) memcpy
  2. m_t1 * delay_one_hot  -> O(B*MD*H) elementwise multiply
  3. add                   -> O(B*MD*H) elementwise add

into a single Triton kernel that does all three with one read+write of the
buffer, two reads (m_t1, delay_one_hot), one write (output). Net: bandwidth
is halved vs naive PyTorch.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    buffer_ptr,
    m_t1_ptr,
    d_oh_ptr,
    out_ptr,
    B,
    MD,
    H,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_md = tl.program_id(1)

    h_off = tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # Source: shift left by 1 — load buffer[b, pid_md+1, :h], zero if pid_md+1 >= MD.
    next_md = pid_md + 1
    valid_src = next_md < MD
    src_offset = pid_b * (MD * H) + next_md * H + h_off
    src_mask = h_mask & valid_src
    src = tl.load(buffer_ptr + src_offset, mask=src_mask, other=0.0)

    # m_t1[b, :h]
    m_offset = pid_b * H + h_off
    m_val = tl.load(m_t1_ptr + m_offset, mask=h_mask, other=0.0)

    # delay_one_hot[b, pid_md, :h]
    d_offset = pid_b * (MD * H) + pid_md * H + h_off
    d_val = tl.load(d_oh_ptr + d_offset, mask=h_mask, other=0.0)

    out_val = src + m_val * d_val
    tl.store(out_ptr + d_offset, out_val, mask=h_mask)


class FusedBufferUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, m_t1, delay_one_hot):
        """
        buffer: (B, MD, H), contiguous fp32
        m_t1: (B, H), contiguous fp32
        delay_one_hot: (B, MD, H), contiguous fp32

        Returns new_buffer (B, MD, H).
        """
        assert buffer.is_cuda and m_t1.is_cuda and delay_one_hot.is_cuda
        assert buffer.is_contiguous() and m_t1.is_contiguous() and delay_one_hot.is_contiguous()

        B, MD, H = buffer.shape
        output = torch.empty_like(buffer)

        BLOCK_H = triton.next_power_of_2(H)
        # Cap BLOCK_H to keep things sensible
        BLOCK_H = max(BLOCK_H, 16)

        grid = (B, MD)
        _fwd_kernel[grid](
            buffer,
            m_t1,
            delay_one_hot,
            output,
            B,
            MD,
            H,
            BLOCK_H=BLOCK_H,
        )

        # Save for backward (note: we don't need `buffer` itself for backward
        # because d_out/d_buffer is a fixed shift pattern, not data-dependent).
        ctx.save_for_backward(m_t1, delay_one_hot)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        m_t1, delay_one_hot = ctx.saved_tensors

        # Forward: out[b, j, h] = (buf[b, j+1, h] if j+1 < MD else 0)
        #                       + m[b, h] * d[b, j, h]
        # ---
        # d_buf[b, k, h] = grad_out[b, k-1, h] if k-1 >= 0 else 0
        #   (shift right by 1, pad zero at start)
        grad_buffer = F.pad(grad_output[:, :-1, :], (0, 0, 1, 0))

        # d_m[b, h] = sum_j grad_out[b, j, h] * d[b, j, h]
        grad_m_t1 = (grad_output * delay_one_hot).sum(dim=1)

        # d_d[b, j, h] = grad_out[b, j, h] * m[b, h]
        grad_delay_one_hot = grad_output * m_t1.unsqueeze(1)

        return grad_buffer, grad_m_t1, grad_delay_one_hot


def fused_buffer_update(buffer, m_t1, delay_one_hot):
    """Convenience wrapper. m_t1 should be (B, H) — if (B, H, 1), squeeze first."""
    return FusedBufferUpdate.apply(buffer, m_t1, delay_one_hot)
