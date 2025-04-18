"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

kernels - Fused self SD attention with small head size

"""

from nks_lang import *


@spec
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref):
    k_t = torch.transpose(k_ref)
    v_t = torch.transpose(v_ref)
    q_k_t = torch.matmul(q_ref, k_t)
    sftmx = torch.softmax(q_k_t, dim=1)
    sftmx_t = torch.transpose(sftmx)
    res_t = torch.matmul(v_t, sftmx_t)
    res = torch.transpose(res_t)
    return res


@sketch
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref):
  pe_in_dt = np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype
  d_head, seqlen = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (d_head, seqlen), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (d_head, seqlen), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen, d_head), f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  assert d_head == 128

  out_ref = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.shared_hbm)


  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128
  assert k_seq_tile_size == 4 * v_seq_tile_size

  v_local = nl.ndarray((v_seq_n_tiles, par_dim(v_seq_tile_size), d_head), dtype=pe_in_dt)
  for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
    v_local[hole()] = hole_op(v_ref[hole()])
  
  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[hole()] = hole_op(q_ref[hole()])

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[hole()] = hole_op(k_ref[hole()])

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles//2):
    reduction_size = 1024
    reduction_tiles = seqlen // reduction_size
    
    local_tp_buf = nl.ndarray((2, k_seq_n_tiles, par_dim(q_seq_tile_size), k_seq_tile_size), dtype=np.float32, buffer=nl.psum)
    attn_res_psum = nl.ndarray((2, par_dim(d_head_tile_size), q_seq_tile_size), dtype=np.float32, buffer=nl.psum)
    sum_local_tp_buf = nl.ndarray((2, par_dim(q_seq_tile_size), k_seq_tile_size), dtype=np.float32, buffer=nl.psum)

    for i_interleave_grp in nl.affine_range(2):
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles): 
        qk_psum[hole()] = nisa.nc_matmul(moving=k_local[hole()], stationary=q_local[hole()])
      
      neg_max_res[hole()] = hole_op(qk_psum[hole()])
      neg_max_res_final[hole()] = nl.negative(neg_max_res[hole()])
      
      for i_exp in nl.affine_range(reduction_tiles):
        exp_res[hole()] = nisa.activation(np.exp, data=qk_psum[hole()], bias=neg_max_res_final[hole()])

      sum_res[hole()] = hole_op(exp_res[hole()])

      for i_v_seq_tile in nl.affine_range(v_seq_n_tiles // 4):
        for i_offset in nl.affine_range(4):
          local_tp_buf[hole()] = hole_op(exp_res[hole()])
          
      for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
        attn_res_psum[hole()] += \
          nisa.nc_matmul(moving=local_tp_buf[hole()], stationary=v_local[hole()])

      attn_res_div[hole()] = nl.divide(attn_res_psum[hole()], sum_res[hole()])
      
      nl.store(out_ref[hole()], value=attn_res_div[hole()])
      
  return out_ref

