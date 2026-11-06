# Copyright (c) 2026, Linear Attention Ablation Study
# Based on MiniRTS (Facebook Research)
#
# Spatial Self-Attention module with two variants:
#   1. Softmax Attention (standard scaled dot-product)  [O(L^2)]
#   2. Linear Attention  (kernelized, O(L) complexity)
#
# Insertion point: After ConvNet output (B, C, H, W), before branch split.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttentionBase(nn.Module):
    """Base class for spatial self-attention."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    def _reshape_for_heads(self, x):
        B, L, C = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _reshape_from_heads(self, x):
        B, _, L, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, self.dim)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.view(B, C, L).transpose(1, 2)
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        out = self._attention(q, k, v)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = out + x_flat
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

    def _attention(self, q, k, v):
        raise NotImplementedError


class SpatialSoftmaxAttention(SpatialAttentionBase):
    """Standard multi-head scaled dot-product attention with softmax."""

    def _attention(self, q, k, v):
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = self._reshape_from_heads(out)
        return out


class SpatialLinearAttention(SpatialAttentionBase):
    """Linear attention using kernel feature maps.
    Based on Katharopoulos et al. 'Transformers are RNNs' (ICML 2020).
    Complexity: O(L) instead of O(L^2).

    kernel options (kernel ablation):
      'elu'  -> phi(x) = elu(x) + 1   (default, used for all reported runs)
      'relu' -> phi(x) = relu(x) + 1  (ReLU kernel)
      'none' -> phi(x) = x            (bare linear attention, no feature map)
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, eps=1e-6, kernel='elu'):
        super().__init__(dim, num_heads, dropout)
        self.eps = eps
        assert kernel in ('elu', 'relu', 'none'), f'unknown kernel: {kernel}'
        self.kernel = kernel

    def _feature_map(self, x):
        if self.kernel == 'elu':
            return F.elu(x) + 1.0
        elif self.kernel == 'relu':
            return F.relu(x) + 1.0
        else:  # 'none'
            return x

    def _attention(self, q, k, v):
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        q = self._feature_map(q)
        k = self._feature_map(k)
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2)
        out = torch.matmul(q, kv)
        normalizer = torch.matmul(q, k_sum.unsqueeze(-1)).squeeze(-1)
        out = out / (normalizer.unsqueeze(-1) + self.eps)
        out = self._reshape_from_heads(out)
        return out


class SpatialWindowedAttention(SpatialAttentionBase):
    """Windowed local attention (Swin-style). window 8x8 -> per-window L=64.

    Complexity: O(L * window_size^2) instead of O(L^2).
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, window_size=8):
        super().__init__(dim, num_heads, dropout)
        self.window_size = window_size

    def _attention(self, q, k, v):
        B, L, C = q.shape
        H = W = int(round(L ** 0.5))
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, \
            'map side %d not divisible by window %d' % (H, ws)
        nh, nw = H // ws, W // ws
        nhead, hd = self.num_heads, self.head_dim

        q = self._reshape_for_heads(q)   # (B, nhead, L, hd)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # (B, nhead, L, hd) -> (B, nhead, nh, ws, nw, ws, hd)
        q = q.view(B, nhead, nh, ws, nw, ws, hd)
        k = k.view(B, nhead, nh, ws, nw, ws, hd)
        v = v.view(B, nhead, nh, ws, nw, ws, hd)
        # -> (B, nhead, nh, nw, ws, ws, hd) -> (B*nhead*nh*nw, ws*ws, hd)
        q = q.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, ws * ws, hd)
        k = k.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, ws * ws, hd)
        v = v.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, ws * ws, hd)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)      # (B*nhead*nh*nw, ws*ws, hd)

        # restore -> (B, nhead, L, hd)
        out = out.reshape(B, nhead, nh, nw, ws, ws, hd)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, nhead, L, hd)
        return self._reshape_from_heads(out)


def _orthogonalize(w):
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
        return torch.linalg.qr(w)[0]
    return torch.qr(w)[0]


class SpatialPerformerAttention(SpatialAttentionBase):
    """Performer (FAVOR+) with orthogonal random features.

    Approximates softmax attention via random feature maps:
        softmax(Q K^T / sqrt(d)) V ~= phi(Q)(phi(K)^T V) / (phi(Q) phi(K)^T 1)
    where phi(x) = exp(W x - ||x||^2/2) / sqrt(m), W orthogonal random.
    Complexity: O(L * m) with m ~= d (=64).
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, num_features=64, ortho=True):
        super().__init__(dim, num_heads, dropout)
        self.num_features = num_features
        self._init_projection(ortho)

    def _init_projection(self, ortho):
        mats = []
        for _ in range(self.num_heads):
            w = torch.randn(self.num_features, self.head_dim)
            if ortho:
                w = _orthogonalize(w)
            mats.append(w)
        self.register_buffer('proj', torch.stack(mats))   # (nhead, m, head_dim)

    def _feature_map(self, x):
        x = x / (self.head_dim ** 0.25)
        proj = torch.matmul(x, self.proj.transpose(-2, -1))   # (B,nhead,L,m)
        norm = (x * x).sum(-1, keepdim=True) * 0.5
        return torch.exp(proj - norm) / (self.num_features ** 0.5)

    def _attention(self, q, k, v):
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        qf = self._feature_map(q)                 # (B,nhead,L,m)
        kf = self._feature_map(k)
        kv = torch.matmul(kf.transpose(-2, -1), v)   # (B,nhead,m,hd)
        z = torch.matmul(qf, kv)                     # (B,nhead,L,hd)
        normalizer = torch.matmul(qf, kf.sum(dim=-2).unsqueeze(-1))  # (B,nhead,L,1)
        out = z / (normalizer + 1e-6)
        return self._reshape_from_heads(out)


class SpatialNystromAttention(SpatialAttentionBase):
    """Nyströmformer with m landmarks (uniformly sampled).

    Approximates softmax attention via the Nyström method:
        softmax(Q K^T/sqrt(d)) ~= K1 (Kc^+) K2,  with m landmarks.
    Complexity: O(L * m), m = 64 landmarks.
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, num_landmarks=64):
        super().__init__(dim, num_heads, dropout)
        self.num_landmarks = num_landmarks

    def _attention(self, q, k, v):
        q = self._reshape_for_heads(q)   # (B,nhead,L,hd)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        B, nhead, L, hd = q.shape
        m = min(self.num_landmarks, L)

        # Landmarks via segment-mean (matches the original Nyströmformer and is
        # more stable / informative than uniform sampling).
        if L % m == 0:
            seg = L // m
            ql = q.reshape(B, nhead, m, seg, hd).mean(dim=3)  # (B,nhead,m,hd)
            kl = k.reshape(B, nhead, m, seg, hd).mean(dim=3)
        else:
            idx = torch.linspace(0, L - 1, m).long().to(q.device)
            ql = q[:, :, idx, :]
            kl = k[:, :, idx, :]

        K1 = torch.softmax(torch.matmul(q, kl.transpose(-2, -1)) / self.scale, dim=-1)   # (B,nhead,L,m)
        Kc = torch.softmax(torch.matmul(ql, kl.transpose(-2, -1)) / self.scale, dim=-1)  # (B,nhead,m,m)
        K2 = torch.softmax(torch.matmul(ql, k.transpose(-2, -1)) / self.scale, dim=-1)   # (B,nhead,m,L)

        # Invert Kc with diagonal loading instead of pinverse. pinverse uses SVD,
        # which is slow and whose backward pass explodes (the root cause of the
        # divergence). Kc is a row-stochastic softmax matrix and is nearly rank-1
        # when landmarks are similar; its smallest singular value can reach 1e-8,
        # making the inverse blow up to 1e15 (loss spikes to 3e15, cmd_type drops
        # to 0.06). A loading of eps=1e-2 keeps Kc_inv bounded around 100.
        eps = 1e-2
        eye = torch.eye(m, device=Kc.device, dtype=Kc.dtype).view(1, 1, m, m)
        Kc_inv = torch.linalg.inv(Kc + eps * eye)   # (B,nhead,m,m)

        out = torch.matmul(K1, torch.matmul(Kc_inv, torch.matmul(K2, v)))
        out = torch.clamp(out, -50.0, 50.0)  # double safety: cap extreme-batch spikes
        return self._reshape_from_heads(out)


def create_attention(attention_type, dim, num_heads=8, dropout=0.0, linear_kernel='elu'):
    if attention_type == 'none' or attention_type is None:
        return None
    elif attention_type == 'softmax':
        return SpatialSoftmaxAttention(dim, num_heads, dropout)
    elif attention_type == 'linear':
        return SpatialLinearAttention(dim, num_heads, dropout, kernel=linear_kernel)
    elif attention_type == 'windowed':
        return SpatialWindowedAttention(dim, num_heads, dropout, window_size=8)
    elif attention_type == 'performer':
        return SpatialPerformerAttention(dim, num_heads, dropout, num_features=64)
    elif attention_type == 'nystrom':
        return SpatialNystromAttention(dim, num_heads, dropout, num_landmarks=64)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


def estimate_complexity(attention_type, L, d, num_heads=8):
    if attention_type == 'softmax':
        flops = 2 * L * L * d + L * L * num_heads
    elif attention_type == 'linear':
        flops = 2 * L * d * d + 4 * L * d
    else:
        flops = 0
    return {
        'type': attention_type, 'L': L, 'd': d,
        'heads': num_heads, 'estimated_flops': flops,
        'complexity_class': 'O(L^2)' if attention_type == 'softmax' else 'O(L)'
    }


if __name__ == '__main__':
    B, C, H, W = 4, 256, 32, 32
    x = torch.randn(B, C, H, W)
    for att in ['softmax', 'linear']:
        m = create_attention(att, C)
        o = m(x)
        assert o.shape == (B, C, H, W), f"{att} shape mismatch: {o.shape}"
        print(f"  [{att}] OK  Params: {sum(p.numel() for p in m.parameters()):,}")
    info_s = estimate_complexity('softmax', 1024, 32, 8)
    info_l = estimate_complexity('linear', 1024, 32, 8)
    print(f"  Softmax FLOPs: {info_s['estimated_flops']:,}")
    print(f"  Linear  FLOPs: {info_l['estimated_flops']:,}")
    print(f"  Speedup: {info_s['estimated_flops']/info_l['estimated_flops']:.1f}x")
