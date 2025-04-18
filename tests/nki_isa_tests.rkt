;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains a bunch of tests to test the semantics of NKI 
;; ISA operations..
;;
;;;;


#lang rosette

(require rosette/lib/synthax)
(require rosette/lib/angelic)
(require racket/pretty)

(require "nki_lang.rkt")
(require "nki_isa.rkt")
(require "nks_ops.rkt")


(define (matmul_ref mtxa mtxb prec)
    (nks::mtx-matmul mtxa mtxb #:prec prec)
)

(define (matmul_test mtxa mtxb prec)
    (define mtxa.t (nki.isa.nc_transpose mtxa #:prec prec))
    (nki.isa.nc_matmul mtxa.t mtxb #:prec prec)
)

(define (verify_matmul_concrete)
    (printf "\n*******************************************")
    (printf "Verification of matmul on concrete inputs...\n")
    ;; Try concrete bitmatrices
    (define a128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define b128 (bv #x00010001000100010001000100010001 128))
    (define prec 8)
    (define mtxa (nks::matrix a128 4 32))
    (define mtxb (nks::matrix b128 4 32))
    (nks::mtx-print mtxa #:name "mtxa" #:prec prec)
    (nks::mtx-print mtxb #:name "mtxb" #:prec prec)
    (define test_result (matmul_ref mtxa mtxb prec))
    (define ref_result (matmul_test mtxa mtxb prec))
    (nks::mtx-print  #:name "(mtxa @ mtxb)" ref_result #:prec 8)
    (nks::mtx-print  #:name "(mtxa.T @ mtxb) mtxa.T? #t" test_result #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_matmul_symbolic)
    (printf "\n*********************************************")
    (printf "\nVerification of matmul on symbolic inputs...\n")
    ;; Try symbolic bitmatrices
    (define-symbolic a128 (bitvector 128))
    (define-symbolic b128 (bitvector 128))
    (define prec 8)
    (define mtxa (nks::matrix a128 4 32))
    (define mtxb (nks::matrix b128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (matmul_ref mtxa mtxb prec))
                            (nks::matrix-vect (matmul_test mtxa mtxb prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (softmax_ref mtx prec)
    (nks::mtx-softmax mtx #:axis 1 #:prec prec)
)

(define (softmax_test mtx prec)
    (define neg_red (nki.isa.tensor_reduce nks::var-max mtx #:axis 1 #:negate #t  #:prec prec))
    (define act (nki.isa.activation nks::var-exp mtx #:bias neg_red #:prec prec))
    (define red (nki.isa.tensor_reduce nks::var-add act #:axis 1 #:prec prec))
    (define rec (nki.isa.reciprocal red #:prec prec))
    (define brdcast (nks::mtx-broadcast rec #:axis 1 #:reps (nks::matrix-rows act)))
    (nki.isa.tensor_tensor act brdcast nks::var-mul #:prec prec)
)

(define (verify_softmax_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of softmax on concrete inputs...\n")
    ;; Try concrete bitmatrices
    (define vect128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (define test_result (softmax_test mtx prec))
    (define ref_result (softmax_ref mtx prec))
    (nks::mtx-print ref_result #:name "softmax reference" #:prec 8)
    (nks::mtx-print test_result #:name "softmax test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_softmax_symbolic)
    (printf "\n**********************************************")
    (printf "\nVerification of softmax on symbolic inputs...\n")
    ;; Try symbolic bitmatrices
    (define-symbolic vect128 (bitvector 128))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (softmax_ref mtx prec))
                            (nks::matrix-vect (softmax_test mtx prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (fused_softmax_test mtx prec)
    (define neg_red (nki.isa.tensor_reduce nks::var-max mtx #:axis 1 #:negate #t  #:prec prec))
    (define-values (act red) (nki.isa.activation_reduce nks::var-exp mtx nks::var-add #:bias neg_red #:prec prec))
    (define rec (nki.isa.reciprocal red #:prec prec))
    (nki.isa.tensor_scalar act nks::var-mul rec #:prec prec)
)

(define (verify_fused_softmax_concrete)
    (printf "\n**************************************************************************")
    (printf "\nVerification of softmax with fused ISA instructions on concrete inputs...\n")
    ;; Try concrete bitmatrices
    (define vect128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (define test_result (fused_softmax_test mtx prec))
    (define ref_result (softmax_ref mtx prec))
    (nks::mtx-print ref_result #:name "softmax reference" #:prec 8)
    (nks::mtx-print test_result #:name "softmax test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_fused_softmax_symbolic)
    (printf "\n**************************************************************************")
    (printf "\nVerification of softmax with fused ISA instructions on symbolic inputs...\n")
    ;; Try symbolic bitmatrices
    (define-symbolic vect128 (bitvector 128))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (softmax_ref mtx prec))
                            (nks::matrix-vect (fused_softmax_test mtx prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (attention_ref q k v prec)
    (define k.T (nks::mtx-transpose k #:prec prec))
    (define q@k.T (nks::mtx-matmul q k.T #:prec prec))
    (define sftmx (nks::mtx-softmax q@k.T #:axis 1 #:prec prec))
    (define v.T (nks::mtx-transpose v #:prec prec))
    (define res (nks::mtx-matmul sftmx v.T #:prec prec))
    res
)

(define (attention_test q k v prec)
    (define q.T (nki.isa.nc_transpose q #:prec prec))
    (define k.T (nki.isa.nc_transpose k #:prec prec))
    (define qk (nki.isa.nc_matmul q.T k.T #:prec prec))
    (define neg_red (nki.isa.tensor_reduce nks::var-max qk #:axis 1 #:negate #t #:prec prec))
    (define-values (act red) (nki.isa.activation_reduce nks::var-exp qk nks::var-add #:bias neg_red #:prec prec))
    (define rec (nki.isa.reciprocal red #:prec prec))
    (define sftmx (nki.isa.tensor_scalar act nks::var-mul rec #:prec prec))
    (define sftmx.T (nki.isa.nc_transpose sftmx #:prec prec))
    (define v.T (nki.isa.nc_transpose v #:prec prec))
    (define res (nki.isa.nc_matmul sftmx.T v.T #:prec prec))
    res
)


(define (verify_attention_concrete)
    (printf "\n************************************************")
    (printf "\nVerification of attention on concrete inputs...\n")
    ;; Try concrete bitmatrices
    (define seqlen 4)
    (define d_head 32)
    (define prec 8)
    (define q (nks::matrix (bv #x000102030405060708090a0b0c0d0e0f 128) seqlen d_head))
    (define k (nks::matrix (bv #x000102030405060708090a0b0c0d0e0f 128) seqlen d_head))
    (define v (nks::matrix (bv #x000102030405060708090a0b0c0d0e0f 128) seqlen d_head))
    (nks::mtx-print q #:name "q" #:prec prec)
    (nks::mtx-print k #:name "k" #:prec prec)
    (nks::mtx-print v #:name "v" #:prec prec)
    (define test_result (attention_ref q k v prec))
    (define ref_result (attention_test q k v prec))
    (nks::mtx-print  #:name "attention reference" ref_result #:prec 8)
    (nks::mtx-print  #:name "attention test" test_result #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_attention_symbolic)
    (printf "\n************************************************")
    (printf "\nVerification of attention on symbolic inputs...\n")
    ;; Try symbolic bitmatrices
    (define seqlen 4)
    (define d_head 32)
    (define prec 8)
    (define-symbolic q128 (bitvector (* seqlen d_head)))
    (define-symbolic k128 (bitvector (* seqlen d_head)))
    (define-symbolic v128 (bitvector (* seqlen d_head)))
    (define q (nks::matrix q128 seqlen d_head))
    (define k (nks::matrix k128 seqlen d_head))
    (define v (nks::matrix v128 seqlen d_head))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (attention_ref q k v prec))
                            (nks::matrix-vect (attention_test q k v prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (rmsnorm_ref mtx weight prec)
    (nks::mtx-rmsnorm mtx weight #:prec prec)
)

(define (rmsnorm_test mtx weight prec)
    (define in_square (nki.isa.tensor_tensor mtx mtx nks::var-mul #:prec prec))
    (define square_sum (nki.isa.tensor_reduce nks::var-add in_square #:axis 1 #:prec prec))
    (define mean (nki.isa.tensor_scalar square_sum nks::var-div (bv (/ (nks::matrix-cols mtx) prec) prec) #:prec prec))
    (define rms_reciprocal (nki.isa.activation nks::var-rsqrt mean #:prec prec))
    (define x_term (nki.isa.tensor_scalar mtx nks::var-mul rms_reciprocal #:prec prec))
    (define bdct_weight (nks::mtx-broadcast weight #:axis 0 #:reps (nks::matrix-rows mtx)))
    (nki.isa.tensor_tensor x_term bdct_weight nks::var-mul #:prec prec)
)

(define (verify_rmsnorm_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of rmsnorm on concrete inputs...\n")
    ;; Try concrete bitmatrices
    (define vect128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define wt32 (bv #x00010203 32))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (define weight (nks::matrix wt32 1 32))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (nks::mtx-print weight #:name "weight" #:prec prec)
    (define ref_result (rmsnorm_ref mtx weight prec))
    (nks::mtx-print ref_result #:name "softmax reference" #:prec 8)
    (define test_result (rmsnorm_test mtx weight prec))
    (nks::mtx-print test_result #:name "softmax test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_rmsnorm_symbolic)
    (printf "\n**********************************************")
    (printf "\nVerification of rmsnorm on symbolic inputs...\n")
    ;; Try symbolic bitmatrices
    (define-symbolic vect128 (bitvector 128))
    (define-symbolic wt32 (bitvector 32))
    (define prec 8)
    (define mtx (nks::matrix vect128 4 32))
    (define weight (nks::matrix wt32 1 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (rmsnorm_ref mtx weight prec))
                            (nks::matrix-vect (rmsnorm_test mtx weight prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (tests)
    (verify_matmul_concrete)
    (verify_matmul_symbolic)

    (verify_softmax_concrete)
    (verify_softmax_symbolic)
    (verify_fused_softmax_concrete)
    (verify_fused_softmax_symbolic)

    (verify_attention_concrete)
    (verify_attention_symbolic)

    (verify_rmsnorm_concrete)
    (verify_rmsnorm_symbolic)
)

(tests)