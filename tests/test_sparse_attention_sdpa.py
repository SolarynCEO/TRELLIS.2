import torch
import torch.nn.functional as F

from trellis2.modules.sparse import VarLenTensor, SparseTensor
from trellis2.modules.sparse import config as sparse_config
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis2.modules.sparse.attention.windowed_attn import sparse_windowed_scaled_dot_product_self_attention


def _manual_varlen_sdpa(qkv: VarLenTensor) -> VarLenTensor:
    outputs = []
    for item in qkv.to_tensor_list():
        q, k, v = item.unbind(dim=1)
        q = q.unsqueeze(0).permute(0, 2, 1, 3)
        k = k.unsqueeze(0).permute(0, 2, 1, 3)
        v = v.unsqueeze(0).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).squeeze(0)
        outputs.append(out)
    return VarLenTensor.from_tensor_list(outputs)


def test_sparse_full_attention_sdpa_matches_manual():
    prev_backend = sparse_config.ATTN
    sparse_config.ATTN = "sdpa"
    try:
        torch.manual_seed(0)
        seq_lens = [3, 2]
        qkv_chunks = [torch.randn((length, 3, 2, 4), dtype=torch.float32) for length in seq_lens]
        qkv = VarLenTensor.from_tensor_list(qkv_chunks)
        out = sparse_scaled_dot_product_attention(qkv)
        ref = _manual_varlen_sdpa(qkv)
        assert isinstance(out, VarLenTensor)
        assert out.feats.shape == ref.feats.shape
        assert torch.allclose(out.feats, ref.feats, atol=1e-5, rtol=1e-4)
    finally:
        sparse_config.ATTN = prev_backend


def test_sparse_windowed_attention_sdpa_runs():
    prev_backend = sparse_config.ATTN
    prev_conv = sparse_config.CONV
    sparse_config.ATTN = "sdpa"
    sparse_config.CONV = "none"
    try:
        torch.manual_seed(1)
        coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
            ],
            dtype=torch.int32,
        )
        feats = torch.randn((coords.shape[0], 3, 2, 4), dtype=torch.float32)
        qkv = SparseTensor(feats=feats, coords=coords)
        out = sparse_windowed_scaled_dot_product_self_attention(qkv, window_size=2)
        assert out.feats.shape == (coords.shape[0], 2, 4)
        assert torch.isfinite(out.feats).all()
    finally:
        sparse_config.ATTN = prev_backend
        sparse_config.CONV = prev_conv
