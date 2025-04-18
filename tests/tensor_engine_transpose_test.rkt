;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; Synthesize weight matrix to perform transpose using tensor engine
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
(require "../nks/nki_isa.rkt")


(define (spec x)
  (define N 8)
  (define rows_b 8)
  (define cols_b (* 8 N))
  (define mtx (nks::matrix x rows_b cols_b))
  (nks::matrix-vect (nks::mtx-transpose mtx #:prec 8))
)


(define (sketch x)
  (define M 8)
  (define N 8)
  (define rows_a 8)
  (define rows_b 8)
  (define cols_a (* 8 M))
  (define cols_b (* 8 N))
  (define-symbolic symbv (bitvector (* rows_a cols_a)))
  (define sym-mtx (nks::matrix symbv rows_a cols_a))
  (define mtx (nks::matrix x rows_b cols_b))
  (nks::matrix-vect (nks::mtx-matmul mtx sym-mtx #:lhs.T? #t #:prec 8))
)


(define-symbolic x (bitvector (* 8 8 8)))

(synthesize
  #:forall (list x)
  #:guarantee (assert (bveq (spec x) (sketch x))))