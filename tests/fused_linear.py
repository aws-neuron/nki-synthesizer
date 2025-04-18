"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

kernels - Fused normalization with linear layers

"""

from nks_lang import *


@nks.spec
def allocated_fused_rms_norm_qkv(hidden, weights):
    mul = torch.square(hidden)
    sum = torch.sum(mul, dim=1)
    rsqrt = torch.rsqrt(sum)
    mul2 = torch.multiply(hidden, rsqrt)
    mult = torch.matmul(mul2, weights)
    return mult


@nks.sketch
def allocated_fused_rms_norm_qkv(hidden, weights):
  seqlen, dim = hidden.shape
  _dim, head_dim = weights.shape

  assert dim <= 8192 and dim % 128 == 0, "Unsupported hidden dimension"
  assert _dim == dim, "Reduction dimension must match"
  assert head_dim <= 512, "Head dimension must be 512 or less"

  out_tensor = nl.ndarray((seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

  pmax, fmax = 128, 512
  M = math.ceil(dim / pmax)
  NUM_TRANSP_TILES = math.ceil(dim / fmax)
  NUM_TILES = math.ceil(seqlen / pmax)
  
  weights_buffer = nl.ndarray((M, par_dim(pmax), fmax), dtype=weights.dtype)
  for m in nl.affine_range(M):
    weights_buffer[hole()] = hole_op(weights[hole()])
  
  for i in nl.affine_range(NUM_TILES):
    in_bufs = hole_op(hidden[hole()])
    act = nisa.activation(nl.square, data=in_bufs)
    square_sum = hole_op(act)
    square_sum = nisa.activation(nl.rsqrt, data=square_sum)
    
    for m in nl.affine_range(NUM_TRANSP_TILES):
      mul_tile = nl.multiply(in_bufs[hole()], square_sum)
      out_tile = hole_op(mul_tile)
    
    res_psum = nl.ndarray((1, par_dim(pmax), fmax), dtype=nl.float32)
    for m in nl.affine_range(M):
      res_psum[hole()] += nisa.nc_matmul(out_tile[hole()], weights_buffer[hole()])
    output_buf = hole_op(res_psum)
    
    nl.store(out_tensor[hole()], value=output_buf)
  return out_tensor

