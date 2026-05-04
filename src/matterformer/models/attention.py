from __future__ import annotations

from matterformer.models.simplicial_attention import (
    SimplicialAttention,
    SimplicialAttentionMask,
    SimplicialFactorizedBias,
    SimplicialLowRankAngleResidual,
    SimplicialLowRankMessageResidual,
    TwoSimplicialAttention,
    _TritonTwoSimplicialAttentionFunction,
    simplicial_attention_torch_from_projected,
)
from matterformer.models.triton_simplicial_attention import (
    TRITON_AVAILABLE,
    SUPPORTED_SIMPLICIAL_PRECISIONS,
    normalize_simplicial_precision,
    triton_block_d_for_head_dim,
    triton_simplicial_attention_backward,
    triton_simplicial_attention_forward,
)

__all__ = [
    "TRITON_AVAILABLE",
    "SUPPORTED_SIMPLICIAL_PRECISIONS",
    "SimplicialAttention",
    "SimplicialAttentionMask",
    "SimplicialFactorizedBias",
    "SimplicialLowRankAngleResidual",
    "SimplicialLowRankMessageResidual",
    "TwoSimplicialAttention",
    "_TritonTwoSimplicialAttentionFunction",
    "normalize_simplicial_precision",
    "simplicial_attention_torch_from_projected",
    "triton_block_d_for_head_dim",
    "triton_simplicial_attention_backward",
    "triton_simplicial_attention_forward",
]
