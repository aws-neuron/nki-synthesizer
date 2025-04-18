;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains the semantics of NKI language operations.
;;
;;;;



#lang rosette

(require "nks_ops.rkt")


(provide (all-defined-out))


(define (nki.lang.add x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-add x #:rhs y #:prec prec))


(define (nki.lang.subtract x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-sub x #:rhs y #:prec prec))


(define (nki.lang.multiply x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-mul x #:rhs y #:prec prec))


(define (nki.lang.divide x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-div x #:rhs y #:prec prec))


(define (nki.lang.maximum x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-max x #:rhs y #:prec prec))


(define (nki.lang.minimum x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-min x #:rhs y #:prec prec))


(define (nki.lang.max x #:axis axis #:prec prec #:mask [mask #f])
    (nks::mtx-reduce nks::var-max x #:axis axis #:prec prec))


(define (nki.lang.min x #:axis axis #:prec prec #:mask [mask #f])
    (nks::mtx-reduce nks::var-min x #:axis axis #:prec prec))


(define (nki.lang.sum x #:axis axis #:prec prec #:mask [mask #f])
    (nks::mtx-reduce nks::var-add x #:axis axis #:prec prec))


(define (nki.lang.prod x #:axis axis #:prec prec #:mask [mask #f])
    (nks::mtx-reduce nks::var-mul x #:axis axis #:prec prec))


;; Combination of two operations which can be "lifted" to generate this instruction.
(define (nki.lang.all x #:axis axis #:prec prec #:mask [mask #f])
    (define tmp (nks::mtx-reduce nks::var-zero? x #:axis axis #:prec prec))
    (nks::mtx-elemwise not tmp #:prec prec))


(define (nki.lang.negative x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-neg x #:prec prec))


(define (nki.lang.exp x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-exp x #:prec prec))


(define (nki.lang.log x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-log x #:prec prec))


(define (nki.lang.cos x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-cos x #:prec prec))


(define (nki.lang.sin x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-sin x #:prec prec))


(define (nki.lang.tan x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-tan x #:prec prec))


(define (nki.lang.tanh x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-tanh x #:prec prec))


(define (nki.lang.arctan x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-arctan x #:prec prec))


(define (nki.lang.sqrt x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-sqrt x #:prec prec))


(define (nki.lang.rsqrt x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-rsqrt x #:prec prec))


(define (nki.lang.relu x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-relu x #:prec prec))


(define (nki.lang.softplus x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-softplus x #:prec prec))


(define (nki.lang.mish x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-mish x #:prec prec))


(define (nki.lang.square x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-mul x #:rhs x #:prec prec))


(define (nki.lang.softmax x #:prec prec #:mask [mask #f])
    (nks::mtx-softmax x #:axis 1 #:prec prec))


(define (nki.lang.matmul x y #:transpose_x? [transpose_x? #f] #:prec prec #:mask [mask #f])
    (nks::mtx-matmul x y #:lhs.T? transpose_x? #:prec prec))


(define (nki.lang.transpose x #:prec prec #:mask [mask #f])
    (nks::mtx-transpose x #:prec prec))


(define (nki.lang.bitwise_and x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-and x #:rhs y #:prec prec))


(define (nki.lang.bitwise_or x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-or x #:rhs y #:prec prec))


(define (nki.lang.bitwise_xor x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-xor x #:rhs y #:prec prec))


(define (nki.lang.invert x #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-not x #:prec prec))


(define (nki.lang.left_shift x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-shl x #:rhs y #:prec prec))


(define (nki.lang.right_shift x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-lshr x #:rhs y #:prec prec))


(define (nki.lang.equal x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-eq x #:rhs y #:prec prec))


(define (nki.lang.not_equal x y #:prec prec #:mask [mask #f])
    (define tmp (nks::mtx-elemwise nks::var-eq x #:rhs y #:prec prec))
    (nks::mtx-elemwise not tmp #:prec prec))


(define (nki.lang.greater x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-gt x #:rhs y #:prec prec))


(define (nki.lang.greater_equal x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-ge x #:rhs y #:prec prec))
    

(define (nki.lang.less x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-lt x #:rhs y #:prec prec))


(define (nki.lang.less_equal x y #:prec prec #:mask [mask #f])
    (nks::mtx-elemwise nks::var-le x #:rhs y #:prec prec))