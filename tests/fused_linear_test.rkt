;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; Fused linear test
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



(define (spec bv_hidden bv_weights )
    (define seqlen 4)
    (define dim 128)
    (define _dim 128)
    (define head_dim 2)
    (define prec 8)
    (define hidden (nks::matrix bv_hidden seqlen (* prec dim)))
    (define weights (nks::matrix bv_weights _dim (* prec head_dim)))

    (define mul (torch.square hidden #:prec 8))
    (define sum (torch.sum mul #:axis 1 #:prec 8))
    (define rsqrt (torch.rsqrt sum #:prec 8))
    (define mul2 (torch.multiply hidden rsqrt #:prec 8))
    (define mult (torch.matmul mul2 weights #:prec 8))

    (nks::matrix-vect mult)
)

(define (sketch bv_hidden bv_weights )
    (define seqlen 4)
    (define dim 128)
    (define _dim 128)
    (define head_dim 2)
    (define prec 8)
    (define pmax 2)
    (define fmax 2)
    (define buffer_depth 2)
    (define M (/ dim pmax))
    (define tile_int (/ seqlen (* buffer_depth pmax)))
    (define num_transp_tiles (/ dim fmax))

    (define hidden (nks::matrix bv_hidden seqlen (* prec dim)))
    (define weights (nks::matrix bv_weights _dim (* prec head_dim)))

    (define weight_buffer
        (apply nks::mtx-rowwise-concat
            (for/list ([i (nks::range M)])
                (nks::mtx-extract weights #:row-idx i #:col-idx 0 #:rows pmax #:cols (* prec head_dim))
            )
        )
    )
    (define result
      (apply nks::mtx-rowwise-concat 
          (for*/list ([i (nks::range tile_int)][j (nks::range buffer_depth)])
              (define hidden_tile (nks::mtx-extract hidden #:row-idx (+ (* i buffer_depth) j) #:col-idx 0 #:rows pmax #:cols (* prec dim)))
              (define act (nki.isa.activation nks::var-square hidden_tile #:prec prec))
              (define square_sum (nks::mtx-reduce nks::var-add act #:axis 1 #:prec prec))
              (define square_sum0 (nki.isa.activation nks::var-rsqrt square_sum #:prec prec))
              (define intres
                  (apply nks::mtx-rowwise-concat 
                      (for/list ([m (nks::range num_transp_tiles)])
                          (define hidden_tile_tile (nks::mtx-extract hidden_tile #:row-idx 0 #:col-idx m #:rows pmax #:cols (* prec fmax)))
                          (define out_tile0 (nki.lang.multiply hidden_tile_tile square_sum0 #:prec prec))
                          (nks::mtx-transpose out_tile0 #:prec prec)
                      )
                  )
              )
              (define dst 
                  (for/list ([m (nks::range M)])
                      (define intres_tile (nks::mtx-extract intres #:row-idx m #:col-idx 0 #:rows pmax #:cols (* prec pmax)))
                      (define weights_tile (nks::mtx-extract weight_buffer #:row-idx m #:col-idx 0 #:rows pmax #:cols (* prec fmax)))
                      (nki.isa.nc_matmul intres_tile weights_tile #:prec prec)
                  )
              )
              (define accum (nks::mtx-accumulate dst #:prec prec))
              accum
          )
      )
    )
    (nks::matrix-vect result)
)


(define (sketch_func params)
  (sketch 
    (vector-ref params 0)
    (vector-ref params 1)
  )
)

(define (spec_func params)
  (spec 
    (vector-ref params 0)
    (vector-ref params 1)
  )
)

(define bitwidth-list (list 4096 2048))

(define (generate-params env)
    (vector (vector-ref env 0) (vector-ref env 1))
)

(define-values (satisfiable? sol? _)
   (cegis-synthesis sketch_func spec_func bitwidth-list generate-params '()))
(define t0 (current-seconds))
(displayln "is Satisfiable?")
(println satisfiable?)
(define t1 (current-seconds))
(- t1 t0)
(if satisfiable? '() (raise 'failed satisfiable?))

