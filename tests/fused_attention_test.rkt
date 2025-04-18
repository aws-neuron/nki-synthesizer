;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; Fused attention test
;;
;;;;


#lang rosette
(require rosette/lib/synthax)
(require rosette/lib/angelic)
(require racket/pretty)
(require racket/serialize)
(require rosette/lib/destruct)
(require rosette/solver/smt/z3)

(require "../nks/nks_ops.rkt")
(require "../nks/cegis.rkt")
(require "../nks/nki_lang.rkt")
(require "../nks/nki_isa.rkt")
(require "../nks/torch.rkt")



(define (spec bv_q_ref bv_k_ref bv_v_ref)
    (define q_ref (nks::matrix bv_q_ref 4 32))
    (define k_ref (nks::matrix bv_k_ref 4 32))
    (define v_ref (nks::matrix bv_v_ref 4 32))
    (nks::mtx-print q_ref #:name "q_ref" #:prec 8)
    (nks::mtx-print k_ref #:name "k_ref" #:prec 8)
    (nks::mtx-print v_ref #:name "v_ref" #:prec 8)

    (define q_t (torch.transpose q_ref #:prec 8))
    (nks::mtx-print q_t #:name "q_t" #:prec 8)
    (define q_k_t (torch.matmul q_t k_ref #:prec 8))
    (nks::mtx-print q_k_t #:name "q_k_t" #:prec 8)
    (define sftmx (torch.softmax q_k_t #:axis 1 #:prec 8))
    (nks::mtx-print sftmx #:name "sftmx" #:prec 8)
    (define v_t (torch.transpose v_ref #:prec 8))
    (nks::mtx-print v_t #:name "v_t" #:prec 8)
    (define res (torch.matmul sftmx v_t #:prec 8))
    (nks::mtx-print res #:name "res" #:prec 8)

    (nks::matrix-vect res)
)


(define (sketch bv_q_ref bv_k_ref bv_v_ref)
    (define seqlen 4)
    (define dhead 4)
    (define prec 8)
    (define pmax 2)
    (define fmax 2)

    (define q_seq_tile_size pmax)
    (define q_seq_n_tiles (/ seqlen q_seq_tile_size))
    (define d_head_tile_size pmax)
    (define d_head_n_tiles (/ dhead d_head_tile_size))
    (define k_seq_tile_size fmax)
    (define k_seq_n_tiles (/ seqlen k_seq_tile_size))
    (define v_seq_tile_size pmax)
    (define v_seq_n_tiles (/ seqlen v_seq_tile_size))
    (define k_seq_v_seq_multipler (/ k_seq_tile_size v_seq_tile_size))

    (define q_ref (nks::matrix bv_q_ref 4 32))
    (define k_ref (nks::matrix bv_k_ref 4 32))
    (define v_ref (nks::matrix bv_v_ref 4 32))

    (define v_local
      (apply nks::mtx-colwise-concat
        (for/list ([i (nks::range d_head_n_tiles)])
          (apply nks::mtx-rowwise-concat
            (for/list ([j (nks::range v_seq_n_tiles)])
              (define tile (nks::mtx-extract v_ref #:row-idx i #:col-idx j #:rows d_head_tile_size #:cols (* prec v_seq_tile_size)))
              (nks::mtx-transpose tile #:prec prec)
            )
          )
        )
      )
    )

    (define result
    (apply nks::mtx-rowwise-concat
        (for/list ([i (nks::range q_seq_n_tiles)])
          (define q_local
            (apply nks::mtx-rowwise-concat
              (for/list ([j (nks::range d_head_n_tiles)])
                (nks::mtx-extract q_ref #:row-idx j #:col-idx i #:rows d_head_tile_size #:cols (* prec q_seq_tile_size))
              )
            )
          )
          (nks::mtx-print q_local #:name "q_local" #:prec prec)

          (define concat-mtx
            (apply nks::mtx-colwise-concat
              (for/list ([j (nks::range k_seq_n_tiles)])
                (define k_local
                  (apply nks::mtx-rowwise-concat
                    (for/list ([k (nks::range d_head_n_tiles)])
                      (nks::mtx-extract k_ref #:row-idx k #:col-idx j #:rows d_head_tile_size #:cols (* prec k_seq_tile_size))
                    )
                  )
                )
                (nks::mtx-print k_local #:name "k_local" #:prec prec)

                (define qk_psum
                  (for/list ([k (nks::range d_head_n_tiles)])
                    (define ktile (nks::mtx-extract k_local #:row-idx k #:col-idx 0 #:rows d_head_tile_size #:cols (* prec k_seq_tile_size)))
                    (define qtile (nks::mtx-extract q_local #:row-idx k #:col-idx 0 #:rows d_head_tile_size #:cols (* prec q_seq_tile_size)))
                    (nks::mtx-print qtile #:name "qtile" #:prec prec)
                    (nks::mtx-print ktile #:name "ktile" #:prec prec)
                    (nks::mtx-print (nki.isa.nc_matmul qtile ktile #:prec prec) #:name "int_qk_psum" #:prec prec)
                    (nki.isa.nc_matmul qtile ktile #:prec prec)
                  )
                )
                (define accum (nks::mtx-accumulate qk_psum #:prec prec))
                (nks::mtx-print accum #:name "accum" #:prec prec)
                accum
              )
            )
          )
          (nks::mtx-print concat-mtx #:name "concat-mtx" #:prec prec)

          (define neg_max_res (nks::mtx-reduce nks::var-max concat-mtx #:axis 1 #:prec prec))
          (define neg_max_res_final (nki.lang.negative neg_max_res #:prec prec))
          
          (define act-mtx
            (apply nks::mtx-colwise-concat
              (for/list ([j (nks::range k_seq_n_tiles)])
                (define tile (nks::mtx-extract concat-mtx #:row-idx 0 #:col-idx j #:rows q_seq_tile_size #:cols (* prec k_seq_tile_size)))
                (nki.isa.activation nks::var-exp tile #:bias neg_max_res_final #:prec prec)
              )
            )
          )
          (nks::mtx-print act-mtx #:name "act-mtx" #:prec prec)
          
          (define sum_res (nks::mtx-reduce nks::var-add act-mtx #:axis 1 #:prec prec))
          (define reciprocal (nki.isa.reciprocal sum_res #:prec prec))
          
          (define intres
            (apply nks::mtx-rowwise-concat
              (for/list ([j (nks::range d_head_n_tiles)])
                  (define res-acc
                    (for/list ([k (nks::range k_seq_n_tiles)])
                      (define tile (nks::mtx-extract act-mtx #:row-idx 0 #:col-idx k #:rows q_seq_tile_size #:cols (* prec k_seq_tile_size)))
                      (define softmax_y (nki.lang.multiply tile reciprocal #:prec prec))
                      (nks::mtx-print softmax_y #:name "softmax_y" #:prec prec)
                      (define mult
                        (for/list ([m (nks::range k_seq_v_seq_multipler)])
                          (define v_tile (nks::mtx-extract v_local #:row-idx (+ (* k k_seq_v_seq_multipler) m) #:col-idx j #:rows v_seq_tile_size #:cols (* prec d_head_tile_size)))
                          (nks::mtx-print v_tile #:name "v_tile" #:prec prec)
                          (define softmax_tile (nks::mtx-extract softmax_y #:row-idx 0 #:col-idx m #:rows q_seq_tile_size #:cols (* prec v_seq_tile_size)))
                          (define softmax_transp (nks::mtx-transpose softmax_tile #:prec prec))
                          (nki.isa.nc_matmul v_tile softmax_transp #:prec prec)
                        )
                      )
                      (define mult_accum (nks::mtx-accumulate mult #:prec prec))
                      (nks::mtx-print mult_accum #:name "mult_accum" #:prec prec)
                      mult_accum
                    )
                  )
                  (define res-acc_accum (nks::mtx-accumulate res-acc #:prec prec))
                  (nks::mtx-print res-acc_accum #:name "res-acc_accum" #:prec prec)
                  res-acc_accum
                )
              )
          )
          (nks::mtx-print intres #:name "intres" #:prec prec)
          (nks::mtx-transpose intres #:prec prec)
        )
      )
    )
    (nks::mtx-print result #:name "result" #:prec prec)
    (nks::matrix-vect result)
    )


(define (sketch_func params)
  (sketch 
    (vector-ref params 0)
    (vector-ref params 1)
    (vector-ref params 2)
  )
)

(define (spec_func params)
  (spec 
    (vector-ref params 0)
    (vector-ref params 1)
    (vector-ref params 2)
  )
)

(define bitwidth-list (list 128 128 128))

(define (generate-params env)
    (vector (vector-ref env 0) (vector-ref env 1) (vector-ref env 2))
)

(define-values (satisfiable? sol? _)
   (cegis-synthesis sketch_func spec_func bitwidth-list generate-params generate-params '()))
(define t0 (current-seconds))
(displayln "is Satisfiable?")
(println satisfiable?)
(define t1 (current-seconds))
(- t1 t0)
(if satisfiable? '() (raise 'failed satisfiable?))