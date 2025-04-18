;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains the semantics of torch language operations.
;;
;;;;



#lang rosette

(require "nks_ops.rkt")


(provide (all-defined-out))


(define (torch.add x y #:prec prec)
    (nks::mtx-elemwise nks::var-add x #:rhs y #:prec prec))


(define (torch.subtract x y #:prec prec)
    (nks::mtx-elemwise nks::var-sub x #:rhs y #:prec prec))


(define (torch.multiply x y #:prec prec)
    (nks::mtx-elemwise nks::var-mul x #:rhs y #:prec prec))


(define (torch.divide x y #:prec prec)
    (nks::mtx-elemwise nks::var-div x #:rhs y #:prec prec))


(define (torch.maximum x y #:prec prec)
    (nks::mtx-elemwise nks::var-max x #:rhs y #:prec prec))


(define (torch.minimum x y #:prec prec)
    (nks::mtx-elemwise nks::var-min x #:rhs y #:prec prec))


(define (torch.max x #:axis axis #:prec prec)
    (nks::mtx-reduce nks::var-max x #:axis axis #:prec prec))


(define (torch.min x #:axis axis #:prec prec)
    (nks::mtx-reduce nks::var-min x #:axis axis #:prec prec))


(define (torch.sum x #:axis axis #:prec prec)
    (nks::mtx-reduce nks::var-add x #:axis axis #:prec prec))


(define (torch.prod x #:axis axis #:prec prec)
    (nks::mtx-reduce nks::var-mul x #:axis axis #:prec prec))


(define (torch.all x #:axis axis #:prec prec)
    (define tmp (nks::mtx-reduce bvzero? x #:axis axis #:prec prec))
    (nks::mtx-elemwise not tmp #:prec prec))


(define (torch.negative x #:prec prec)
    (nks::mtx-elemwise nks::var-neg x #:prec prec))


(define (torch.exp x #:prec prec)
    (nks::mtx-elemwise nks::var-exp x #:prec prec))


(define (torch.log x #:prec prec)
    (nks::mtx-elemwise nks::var-log x #:prec prec))


(define (torch.cos x #:prec prec)
    (nks::mtx-elemwise nks::var-cos x #:prec prec))


(define (torch.sin x #:prec prec)
    (nks::mtx-elemwise nks::var-sin x #:prec prec))


(define (torch.tan x #:prec prec)
    (nks::mtx-elemwise nks::var-tan x #:prec prec))


(define (torch.tanh x #:prec prec)
    (nks::mtx-elemwise nks::var-tanh x #:prec prec))


(define (torch.arctan x #:prec prec)
    (nks::mtx-elemwise nks::var-arctan x #:prec prec))


(define (torch.sqrt x #:prec prec)
    (nks::mtx-elemwise nks::var-sqrt x #:prec prec))


(define (torch.rsqrt x #:prec prec)
    (nks::mtx-elemwise nks::var-rsqrt x #:prec prec))


(define (torch.relu x #:prec prec)
    (nks::mtx-elemwise nks::var-relu x #:prec prec))


(define (torch.softplus x #:prec prec)
    (nks::mtx-elemwise nks::var-softplus x #:prec prec))


(define (torch.mish x #:prec prec)
    (nks::mtx-elemwise nks::var-mish x #:prec prec))


(define (torch.square x #:prec prec)
    (nks::mtx-elemwise nks::var-square x #:prec prec))


(define (torch.softmax x #:axis axis #:prec prec)
    (nks::mtx-softmax x #:axis axis #:prec prec))


(define (torch.matmul x y #:prec prec)
    (nks::mtx-matmul x y #:prec prec))


(define (torch.transpose x #:prec prec)
    (nks::mtx-transpose x #:prec prec))


(define (torch.bitwise_and x y #:prec prec)
    (nks::mtx-elemwise nks::var-and x #:rhs y #:prec prec))


(define (torch.bitwise_or x y #:prec prec)
    (nks::mtx-elemwise nks::var-or x #:rhs y #:prec prec))


(define (torch.bitwise_xor x y #:prec prec)
    (nks::mtx-elemwise nks::var-xor x #:rhs y #:prec prec))


(define (torch.invert x #:prec prec)
    (nks::mtx-elemwise nks::var-not x #:prec prec))


(define (torch.left_shift x y #:prec prec)
    (nks::mtx-elemwise nks::var-shl x #:rhs y #:prec prec))


(define (torch.right_shift x y #:prec prec)
    (nks::mtx-elemwise nks::var-lshr x #:rhs y #:prec prec))


(define (torch.equal x y #:prec prec)
    (nks::mtx-elemwise nks::var-eq x #:rhs y #:prec prec))


(define (torch.not_equal x y #:prec prec)
     (define tmp (nks::mtx-elemwise nks::var-eq x #:rhs y #:prec prec))
    (nks::mtx-elemwise not tmp #:prec prec))


(define (torch.greater x y #:prec prec)
    (nks::mtx-elemwise nks::var-gt x #:rhs y #:prec prec))


(define (torch.greater_equal x y #:prec prec)
    (nks::mtx-elemwise nks::var-ge x #:rhs y #:prec prec))
    

(define (torch.less x y #:prec prec)
    (nks::mtx-elemwise nks::var-lt x #:rhs y #:prec prec))


(define (torch.less_equal x y #:prec prec)
    (nks::mtx-elemwise nks::var-le x #:rhs y #:prec prec))

