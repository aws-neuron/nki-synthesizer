;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains a bunch of tests to test the semantics of general 
;; matrix operations.
;;
;;;;


#lang rosette

(require rosette/lib/synthax)
(require rosette/lib/angelic)
(require racket/pretty)

(require "nks_ops.rkt")


(define (matmul_ref mtx-a mtx-b prec)
    (nks::mtx-matmul mtx-a mtx-b #:prec prec)
)

(define (matmul_test mtx-a mtx-b prec)
    (define mtx-a.t (nks::mtx-transpose mtx-a #:prec prec))
    (nks::mtx-matmul mtx-a.t mtx-b #:lhs.T? #t #:prec prec)
)

(define (verify_matmul_concrete)
    (printf "\n*******************************************")
    (printf "Verification of matmul on concrete inputs...\n")
    ;; Try concrete matrices
    (define a128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define b128 (bv #x00010001000100010001000100010001 128))
    (define prec 8)
    (define mtx-a (nks::matrix a128 4 32))
    (define mtx-b (nks::matrix b128 4 32))
    (nks::mtx-print mtx-a #:name "mtx-a" #:prec prec)
    (nks::mtx-print mtx-b #:name "mtx-b" #:prec prec)
    (define ref_result (matmul_test mtx-a mtx-b prec))
     (define test_result (matmul_ref mtx-a mtx-b prec))
    (nks::mtx-print  #:name "(nks::mtx-a @ mtx-b)" ref_result #:prec 8)
    (nks::mtx-print  #:name "(nks::mtx-a.T @ mtx-b) mtx-a.T? #t" test_result #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_matmul_symbolic)
    (printf "\n*********************************************")
    (printf "\nVerification of matmul on symbolic inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic a128 (bitvector 128))
    (define-symbolic b128 (bitvector 128))
    (define prec 8)
    (define mtx-a (nks::matrix a128 4 32))
    (define mtx-b (nks::matrix b128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (matmul_ref mtx-a mtx-b prec))
                            (nks::matrix-vect (matmul_test mtx-a mtx-b prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (softmax_ref mtx prec)
    (nks::mtx-softmax mtx #:axis 1 #:prec prec)
)

(define (softmax_test mtx prec)
    (define red1 (nks::mtx-reduce nks::var-max mtx #:axis 1 #:prec prec))
    (define neg_red1 (nks::mtx-elemwise nks::var-neg red1 #:prec prec))
    (define act (nks::mtx-activation nks::var-exp mtx #:bias neg_red1 #:prec prec))
    (define red2 (nks::mtx-reduce nks::var-add act #:axis 1 #:prec prec))
    (define div (nks::mtx-elemwise nks::var-div (bv 1 prec) #:rhs red2 #:prec prec))
    (nks::mtx-elemwise nks::var-mul act #:rhs div #:prec prec)
)

(define (verify_softmax_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of softmax on concrete inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
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
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (softmax_ref mtx prec))
                            (nks::matrix-vect (softmax_test mtx prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (softmax_axis0_ref mtx prec)
    (nks::mtx-softmax mtx #:axis 1 #:prec prec)
)

(define (softmax_axis0_test mtx prec)
    (define mtx.t (nks::mtx-transpose mtx #:prec prec))
    (define sft (nks::mtx-softmax mtx.t #:axis 0 #:prec prec))
    (nks::mtx-transpose sft #:prec prec)
)

(define (verify_softmax_axis0_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of softmax axis 0 on concrete inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (define ref_result (softmax_axis0_ref mtx prec))
    (nks::mtx-print ref_result #:name "softmax reference" #:prec 8)
    (define test_result (softmax_axis0_test mtx prec))
    (nks::mtx-print test_result #:name "softmax test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_softmax_axis0_symbolic)
    (printf "\n**********************************************")
    (printf "\nVerification of softmax axis 0 on symbolic inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (softmax_axis0_ref mtx prec))
                            (nks::matrix-vect (softmax_axis0_test mtx prec))))))
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
    (define q.T (nks::mtx-transpose q #:prec prec))
    (define k.T (nks::mtx-transpose k #:prec prec))
    (define qk (nks::mtx-matmul q.T k.T #:prec prec #:lhs.T? #t))
    (define red1 (nks::mtx-reduce nks::var-max qk #:axis 1 #:prec prec))
    (define neg_red1 (nks::mtx-elemwise nks::var-neg red1 #:prec prec))
    (define act (nks::mtx-activation nks::var-exp qk #:bias neg_red1 #:prec prec))
    (define red3 (nks::mtx-reduce nks::var-add act #:axis 1 #:prec prec))
    (define div (nks::mtx-elemwise nks::var-div (bv 1 prec) #:rhs red3 #:prec prec))
    (define sftmx (nks::mtx-elemwise nks::var-mul act #:rhs div #:prec prec))
    (define v.T (nks::mtx-transpose v #:prec prec))
    (define sftmx.T (nks::mtx-transpose sftmx #:prec prec))
    (define res (nks::mtx-matmul sftmx.T v.T #:prec prec #:lhs.T? #t))
    res
)

(define (verify_attention_concrete)
    (printf "\n************************************************")
    (printf "\nVerification of attention on concrete inputs...\n")
    ;; Try concrete matrices
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
    ;; Try symbolic matrices
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


(define (elemwise_scalar_broadcast_ref op mtx scalar prec)
    (define cols (nks::matrix-cols mtx))
    (define rows (nks::matrix-rows mtx))
    (define vect (nks::matrix-vect mtx))
    (define result 
        (apply concat
            (for*/list ([i (nks::range rows)][j (nks::range cols prec)])
                (define low (+ j (* i cols)))
                (define high (+ low prec -1))
                (op (extract high low vect) scalar))))
    (nks::matrix result rows cols)
)

(define (elemwise_scalar_broadcast_test op mtx scalar prec)
    (nks::mtx-elemwise op mtx #:rhs scalar #:prec prec)
)

(define (verify_elemwise_scalar_broadcast_concrete)
    (printf "\n*******************************************************************")
    (printf "\nVerification of elementwise scalar broadcast on concrete inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define scalar (bv #x02 8))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (printf "scalar:\n")
    (print scalar)
    (printf "\n")
    (define ref_result (elemwise_scalar_broadcast_ref nks::var-sub mtx scalar prec))
    (define test_result (elemwise_scalar_broadcast_test nks::var-sub mtx scalar prec))
    (nks::mtx-print ref_result #:name "elemwise reference" #:prec 8)
    (nks::mtx-print test_result #:name "elemwise test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_scalar_broadcast_symbolic)
    (printf "\n*******************************************************************")
    (printf "\nVerification of elementwise scalar broadcast on symbolic inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define-symbolic scalar (bitvector 8))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (elemwise_scalar_broadcast_test nks::var-sub mtx scalar prec))
                            (nks::matrix-vect (elemwise_scalar_broadcast_ref nks::var-sub mtx scalar prec)))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (elemwise_scalar_broadcast_comm_ref op scalar mtx prec)
    (define cols (nks::matrix-cols mtx))
    (define rows (nks::matrix-rows mtx))
    (define vect (nks::matrix-vect mtx))
    (define result 
        (apply concat
            (for*/list ([i (nks::range rows)][j (nks::range cols prec)])
                (define low (+ j (* i cols)))
                (define high (+ low prec -1))
                (op scalar (extract high low vect)))))
    (nks::matrix result rows cols)
)

(define (elemwise_scalar_broadcast_comm_test op scalar mtx prec)
    (nks::mtx-elemwise op scalar #:rhs mtx #:prec prec)
)

(define (verify_elemwise_scalar_broadcast_comm_concrete)
    (printf "\n********************************************************************************************")
    (printf "\nVerification of elementwise scalar broadcast on concrete inputs after commutating inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define scalar (bv #x02 8))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (printf "scalar:\n")
    (print scalar)
    (printf "\n")
    (define ref_result (elemwise_scalar_broadcast_comm_ref nks::var-sub scalar mtx prec))
    (define test_result (elemwise_scalar_broadcast_comm_test nks::var-sub scalar mtx prec))
    (nks::mtx-print ref_result #:name "elemwise reference" #:prec 8)
    (nks::mtx-print test_result #:name "elemwise test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_scalar_broadcast_comm_symbolic)
    (printf "\n********************************************************************************************")
    (printf "\nVerification of elementwise scalar broadcast on symbolic inputs after commutating inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define-symbolic scalar (bitvector 8))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (elemwise_scalar_broadcast_comm_test nks::var-sub scalar mtx prec))
                            (nks::matrix-vect (elemwise_scalar_broadcast_comm_ref nks::var-sub scalar mtx prec)))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (elemwise_vect_broadcast_ref op mtx vect prec)
    (define cols (nks::matrix-cols mtx))
    (define rows (nks::matrix-rows mtx))
    (define vecta (nks::matrix-vect mtx))
    (define vectb (nks::matrix-vect vect))
    (define result 
        (apply concat
            (for*/list ([i (nks::range rows)][j (nks::range cols prec)])
                (define lowa (+ j (* i cols)))
                (define higha (+ lowa prec -1))
                (define lowb (* i prec))      
                (define highb (+ lowb prec -1))
                (op (extract higha lowa vecta) (extract highb lowb vectb)))))
    (nks::matrix result rows cols)
)

(define (elemwise_vect_broadcast_test op mtx vect prec)
    (nks::mtx-elemwise op mtx #:rhs vect #:prec prec)
)

(define (verify_elemwise_vector_broadcast_concrete)
    (printf "\n*******************************************************************")
    (printf "\nVerification of elementwise vector broadcast on concrete inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define vect (bv #x00010203 32))
    (define mtxvect (nks::matrix vect 4 8))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (nks::mtx-print mtxvect #:name "mtxvect" #:prec prec)
    (define ref_result (elemwise_vect_broadcast_ref nks::var-sub mtx mtxvect prec))
    (define test_result (elemwise_vect_broadcast_test nks::var-sub mtx mtxvect prec))
    (nks::mtx-print ref_result #:name "elemwise reference" #:prec 8)
    (nks::mtx-print test_result #:name "elemwise test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_vector_broadcast_symbolic)
    (printf "\n*******************************************************************")
    (printf "\nVerification of elementwise vector broadcast on symbolic inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define-symbolic vect (bitvector 32))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define mtxvect (nks::matrix vect 4 8))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (elemwise_vect_broadcast_ref nks::var-sub mtx mtxvect prec))
                            (nks::matrix-vect (elemwise_vect_broadcast_test nks::var-sub mtx mtxvect prec)))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (elemwise_vect_broadcast_comm_ref op mtx vect prec)
    (define cols (nks::matrix-cols mtx))
    (define rows (nks::matrix-rows mtx))
    (define vecta (nks::matrix-vect mtx))
    (define vectb (nks::matrix-vect vect))
    (define result 
        (apply concat
            (for*/list ([i (nks::range rows)][j (nks::range cols prec)])
                (define lowa (+ j (* i cols)))
                (define higha (+ lowa prec -1))
                (define lowb (* i prec))      
                (define highb (+ lowb prec -1))
                (op (extract highb lowb vectb) (extract higha lowa vecta)))))
    (nks::matrix result rows cols)
)

(define (elemwise_vect_broadcast_comm_test op mtx vect prec)
    (nks::mtx-elemwise op vect #:rhs mtx #:prec prec)
)

(define (verify_elemwise_vector_broadcast_comm_concrete)
    (printf "\n********************************************************************************************")
    (printf "\nVerification of elementwise vector broadcast on concrete inputs after commutating inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define vect (bv #x00010203 32))
    (define mtxvect (nks::matrix vect 4 8))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (nks::mtx-print mtxvect #:name "mtxvect" #:prec prec)
    (define ref_result (elemwise_vect_broadcast_comm_ref nks::var-sub mtx mtxvect prec))
    (define test_result (elemwise_vect_broadcast_comm_test nks::var-sub mtx mtxvect prec))
    (nks::mtx-print ref_result #:name "elemwise reference" #:prec 8)
    (nks::mtx-print test_result #:name "elemwise test" #:prec 8)
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_vector_broadcast_comm_symbolic)
    (printf "\n********************************************************************************************")
    (printf "\nVerification of elementwise vector broadcast on symbolic inputs after commutating inputs...\n")
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define-symbolic vect (bitvector 32))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define mtxvect (nks::matrix vect 4 8))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (elemwise_vect_broadcast_comm_ref nks::var-sub mtx mtxvect prec))
                            (nks::matrix-vect (elemwise_vect_broadcast_comm_test nks::var-sub mtx mtxvect prec)))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (elemwise_scalar_ref op scalar1 scalar2 prec)
    (op scalar1 scalar2)
)

(define (elemwise_scalar_test op scalar1 scalar2 prec)
    (nks::mtx-elemwise op scalar1 #:rhs scalar2 #:prec prec)
)

(define (verify_elemwise_scalar_concrete)
    (printf "\n*********************************************************")
    (printf "\nVerification of elementwise scalar on concrete inputs...\n")
    ;; Try concrete scalars
    (define prec 32)
    (define scalar1 (bv #x00010203 32))
    (define scalar2 (bv #x04050607 32))
    (printf "scalar1:\n")
    (print scalar1)
    (printf "\n")
    (printf "scalar2:\n")
    (print scalar2)
    (printf "\n")
    (define ref_result (elemwise_scalar_ref nks::var-sub scalar1 scalar2 prec))
    (define test_result (elemwise_scalar_test nks::var-sub scalar1 scalar2 prec))
    (printf "ref_result:\n")
    (print ref_result)
    (printf "\n")
    (printf "test_result:\n")
    (print test_result)
    (printf "\n")
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_scalar_symbolic)
    (printf "\n*********************************************************")
    (printf "\nVerification of elementwise scalar on symbolic inputs...\n")
    ;; Try symbolic scalars
    (define-symbolic scalar1 (bitvector 32))
    (define-symbolic scalar2 (bitvector 32))
    (define prec 32)
    (define sol?
        (verify (assert (nks::var-eq 
                         (elemwise_scalar_ref nks::var-sub scalar1 scalar2 prec)
                         (elemwise_scalar_test nks::var-sub scalar1 scalar2 prec))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (elemwise_scalar_comm_ref op scalar1 scalar2 prec)
    (op scalar2 scalar1)
)

(define (elemwise_scalar_comm_test op scalar1 scalar2 prec)
    (nks::mtx-elemwise op scalar2 #:rhs scalar1 #:prec prec)
)

(define (verify_elemwise_scalar_comm_concrete)
    (printf "\n**********************************************************************************")
    (printf "\nVerification of elementwise scalar on concrete inputs after commutating inputs...\n")
    ;; Try concrete scalars
    (define prec 32)
    (define scalar1 (bv #x00010203 32))
    (define scalar2 (bv #x04050607 32))
    (printf "scalar1:\n")
    (print scalar1)
    (printf "\n")
    (printf "scalar2:\n")
    (print scalar2)
    (printf "\n")
    (define ref_result (elemwise_scalar_comm_ref nks::var-sub scalar1 scalar2 prec))
    (define test_result (elemwise_scalar_comm_test nks::var-sub scalar1 scalar2 prec))
    (printf "ref_result:\n")
    (print ref_result)
    (printf "\n")
    (printf "test_result:\n")
    (print test_result)
    (printf "\n")
    (printf "equal?:\n")
    (pretty-print (equal? test_result ref_result))
    (printf "\n")
)

(define (verify_elemwise_scalar_comm_symbolic)
    (printf "\n**********************************************************************************")
    (printf "\nVerification of elementwise scalar on symbolic inputs after commutating inputs...\n")
    ;; Try symbolic scalars
    (define-symbolic scalar1 (bitvector 32))
    (define-symbolic scalar2 (bitvector 32))
    (define prec 32)
    (define sol?
        (verify (assert (nks::var-eq 
                         (elemwise_scalar_comm_ref nks::var-sub scalar1 scalar2 prec)
                         (elemwise_scalar_comm_test nks::var-sub scalar1 scalar2 prec))))
    )
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (rmsnorm_ref mtx weight prec)
    (nks::mtx-rmsnorm mtx weight #:prec prec)
)

(define (rmsnorm_test mtx weight prec)
    (define in_square (nks::mtx-elemwise nks::var-mul mtx #:rhs mtx #:prec prec))
    (define square_sum (nks::mtx-reduce nks::var-add in_square #:axis 1 #:prec prec))
    (define mean (nks::mtx-elemwise nks::var-div square_sum #:rhs (bv (/ (nks::matrix-cols mtx) prec) prec) #:prec prec))
    (define rms_reciprocal (nks::mtx-elemwise nks::var-rsqrt mean #:prec prec))
    (define x_term (nks::mtx-elemwise nks::var-mul mtx #:rhs rms_reciprocal #:prec prec))
    (define bdct_weight (nks::mtx-broadcast weight #:axis 0 #:reps (nks::matrix-rows mtx)))
    (nks::mtx-elemwise nks::var-mul x_term #:rhs bdct_weight #:prec prec)
)

(define (verify_rmsnorm_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of rmsnorm on concrete inputs...\n")
    ;; Try concrete matrices
    (define var128 (bv #x000102030405060708090a0b0c0d0e0f 128))
    (define wt32 (bv #x00010203 32))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
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
    ;; Try symbolic matrices
    (define-symbolic var128 (bitvector 128))
    (define-symbolic wt32 (bitvector 32))
    (define prec 8)
    (define mtx (nks::matrix var128 4 32))
    (define weight (nks::matrix wt32 1 32))
    (define sol?
        (verify (assert (nks::var-eq 
                            (nks::matrix-vect (rmsnorm_ref mtx weight prec))
                            (nks::matrix-vect (rmsnorm_test mtx weight prec))))))
    (printf "equal?:\n")
    (pretty-print (unsat? sol?))
    (printf "\n")
)


(define (verify_maxpool_concrete)
    (printf "\n**********************************************")
    (printf "\nVerification of max pool on concrete inputs...\n")
    ;; Try concrete matrices
    (define var256 (bv #x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 256))
    (define prec 8)
    (define mtx (nks::matrix var256 2 128))
    (nks::mtx-print mtx #:name "mtx" #:prec prec)
    (define result (nks::mtx-window-reduce nks::var-max mtx #:window (cons 2 16) 
                                            #:strides (cons 2 16) #:shape (cons 4 32) #:prec 8))
    (nks::mtx-print result #:name "result" #:prec 8)
    (define result2 (nks::mtx-window-reduce nks::var-max mtx #:window (cons 2 16) 
                                            #:strides (cons 2 8) #:shape (cons 4 32) #:prec 8))
    (nks::mtx-print result2 #:name "result2" #:prec 8)

    (define result3 (nks::mtx-window-reduce nks::var-max mtx #:window (cons 1 128) 
                                            #:strides (cons 1 1) #:shape (cons 1 128) #:prec 8))
    (nks::mtx-print result3 #:name "result3" #:prec 8)
    (define reg_result (nks::mtx-reduce nks::var-max mtx #:axis 1 #:prec 8))
    (nks::mtx-print reg_result #:name "reg_result" #:prec 8)
)


(define (tests)
    (verify_matmul_concrete)
    (verify_matmul_symbolic)

    (verify_softmax_concrete)
    (verify_softmax_symbolic)

    (verify_softmax_axis0_concrete)
    (verify_softmax_axis0_symbolic)

    (verify_attention_concrete)
    (verify_attention_symbolic)

    (verify_elemwise_scalar_broadcast_concrete)
    (verify_elemwise_scalar_broadcast_symbolic)
    (verify_elemwise_scalar_broadcast_comm_concrete)
    (verify_elemwise_scalar_broadcast_comm_symbolic)

    (verify_elemwise_vector_broadcast_concrete)
    (verify_elemwise_vector_broadcast_symbolic)
    (verify_elemwise_vector_broadcast_comm_concrete)
    (verify_elemwise_vector_broadcast_comm_symbolic)

    (verify_elemwise_scalar_concrete)
    (verify_elemwise_scalar_symbolic)
    (verify_elemwise_scalar_comm_concrete)
    (verify_elemwise_scalar_comm_symbolic)

    (verify_rmsnorm_concrete)
    (verify_rmsnorm_symbolic)

    (verify_maxpool_concrete)
)

(tests)