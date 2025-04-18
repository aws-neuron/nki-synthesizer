"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

kernels - Fused self attention forward pass

"""

from nks_lang import *


@nks.spec
def fused_self_attn_fwd(q_ref, k_ref, v_ref):
    k_t = torch.transpose(k_ref)
    q_k_t = torch.matmul(q_ref, k_t)
    sftmx = torch.softmax(q_k_t, dim=1)
    res = torch.matmul(sftmx, v_ref)
    return res


@nks.sketch
def fused_self_attn_fwd(q_ref, k_ref, v_ref):
    kernel_dtype = q_ref.dtype
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype
    d_head, seqlen = q_ref.shape
    assert tuple(k_ref.shape) == (d_head, seqlen), 'Input shape mismatch!'
    assert tuple(v_ref.shape) == (d_head, seqlen), 'Input shape mismatch!'
    assert tuple(q_ref.shape) == (d_head, seqlen), 'Input shape mismatch!'
    assert d_head <= 128 or (d_head % 128 == 0), 'd_head must be <= 128 or divisible by 128!'

    out_ref = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
    d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)
    v_seq_n_tiles, v_seq_tile_size = div_ceil(seqlen, 128), 128
    if seqlen >= 512:
        k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
    else:
        k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128

    k_seq_v_seq_multipler = k_seq_tile_size // v_seq_tile_size

    trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=kernel_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
            trans_v[hole()] = hole_op(v_ref[hole()])

    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local = nl.ndarray((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
            q_local[hole()] = hole_op(q_ref[hole()])

        neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), buffer=nl.sbuf, dtype=kernel_dtype)
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            k_local = nl.ndarray((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
            for i_d_head_tile in nl.affine_range(d_head_n_tiles):
                k_local[hole()] = hole_op(k_ref[hole()])

        qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=np.float32, buffer=nl.psum)
        for i_d_head_tile in nl.affine_range(d_head_n_tiles): 
            qk_psum[hole()] += nisa.nc_matmul(q_local[hole()], k_local[hole()])
        
        neg_max_res = hole_op(qk_psum)  
        neg_max_res_final = nl.negative(neg_max_res)

        softmax_numerator = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            softmax_numerator[hole()] = \
                nisa.activation(np.exp, data=qk_psum[hole()], bias=neg_max_res_final[hole()], scale=1.0)

        sum_res = hole_op(softmax_numerator)
        reciprocal = nisa.reciprocal(sum_res)

        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
            attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size), dtype=np.float32, buffer=nl.psum)
            for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
                softmax_y = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf) 
                softmax_y[hole()] = nl.multiply(softmax_numerator[hole()], reciprocal[hole()])

                for i_v_seq_tile in nl.affine_range(k_seq_v_seq_multipler):
                    trans_softmax_res = hole_op(softmax_y[hole()])
                    attn_res_psum[hole()] += nisa.nc_matmul(stationary=trans_softmax_res, moving=trans_v[hole()])
        
            nl.store(out_ref[hole()], value=attn_res_psum[hole()])
    return out_ref
