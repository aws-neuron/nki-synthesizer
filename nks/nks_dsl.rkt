;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains definitions of a bunch of structs that 
;; represent abstractions in an internal DSL for NKS that is 
;; used for synthesis and verification. We also define interpreters 
;; that translates the abstractions of this DSL to their semantics.
;;
;;;;


#lang rosette

(require rosette/lib/destruct)

(require "nks_ops.rkt")
(require "nki_isa.rkt")
(require "nki_lang.rkt")


(provide (all-defined-out))


;; First order operations that are composed together for tensor ops.
(struct nks::var (a) #:transparent #:mutable)
(struct nks::pack2 (a b) #:transparent #:mutable)
(struct nks::pack4 (a b c d) #:transparent #:mutable)
(struct nks::pack6 (a b c d e f) #:transparent #:mutable)
(struct nks::pack8 (a b c d e f g h) #:transparent #:mutable)
(struct nks::add (a b) #:transparent #:mutable)
(struct nks::sub (a b) #:transparent #:mutable)
(struct nks::mul (a b) #:transparent #:mutable)
(struct nks::div (a b) #:transparent #:mutable)
(struct nks::min (a b) #:transparent #:mutable)
(struct nks::max (a b) #:transparent #:mutable)
(struct nks::neg (a) #:transparent #:mutable)
(struct nks::exp (a len) #:transparent #:mutable)
(struct nks::log (a len) #:transparent #:mutable)
(struct nks::sin (a len) #:transparent #:mutable)
(struct nks::cos (a len) #:transparent #:mutable)
(struct nks::tan (a len) #:transparent #:mutable)
(struct nks::tanh (a len) #:transparent #:mutable)
(struct nks::arctan (a len) #:transparent #:mutable)
(struct nks::softplus (a len) #:transparent #:mutable)
(struct nks::relu (a len) #:transparent #:mutable)
(struct nks::mish (a len) #:transparent #:mutable)
(struct nks::sqrt (a len) #:transparent #:mutable)
(struct nks::rsqrt (a len) #:transparent #:mutable)
(struct nks::square (a len) #:transparent #:mutable)

;; Basic NKS tensor operations. These are the operations that 
;; the first order operations are "lifted" to.
(struct nks::mtx::transpose (a prec) #:transparent #:mutable)
(struct nks::mtx::matmul (a b rhs.T? prec) #:transparent #:mutable)
(struct nks::mtx::broadcast (a axis reps) #:transparent #:mutable)
(struct nks::mtx::reduce (op a axis prec) #:transparent #:mutable)
(struct nks::mtx::elemwise (op a b prec) #:transparent #:mutable)
(struct nks::mtx::activation (op a b scale prec) #:transparent #:mutable)
(struct nks::mtx::rmsnorm (a b prec) #:transparent #:mutable)
(struct nks::mtx::softmax (a b prec) #:transparent #:mutable)

;; Interpreter for NKS abstractions
(define (nks::interpret program)
  (destruct program
    [(nks::var a) a]
    [(nks::pack2 a b) (concat (nks::interpret a) (nks::interpret b))]
    [(nks::pack4 a b c d) (concat (nks::interpret a) (nks::interpret b) (nks::interpret c) (nks::interpret d))]
    [(nks::pack6 a b c d e f) (concat (nks::interpret a) (nks::interpret b) (nks::interpret c) 
                                      (nks::interpret d) (nks::interpret e) (nks::interpret f))]
    [(nks::pack8 a b c d e f g h) (concat (nks::interpret a) (nks::interpret b) (nks::interpret c) 
                                          (nks::interpret d) (nks::interpret e) (nks::interpret f)
                                          (nks::interpret g) (nks::interpret h))]
    [(nks::add a b) (nks::var-add (nks::interpret a) (nks::interpret b))]
    [(nks::sub a b) (nks::var-sub (nks::interpret a) (nks::interpret b))]
    [(nks::mul a b) (nks::var-mul (nks::interpret a) (nks::interpret b))]
    [(nks::div a b) (nks::var-div (nks::interpret a) (nks::interpret b))]
    [(nks::min a b) (nks::var-min (nks::interpret a) (nks::interpret b))]
    [(nks::max a b) (nks::var-max (nks::interpret a) (nks::interpret b))]
    [(nks::neg a) (nks::var-neg (nks::interpret a))]
    [(nks::exp a len) (nks::var-exp (nks::interpret a) #:len len)]
    [(nks::log a len) (nks::var-log (nks::interpret a) #:len len)]
    [(nks::sin a len) (nks::var-sin (nks::interpret a) #:len len)]
    [(nks::cos a len) (nks::var-cos (nks::interpret a) #:len len)]
    [(nks::tan a len) (nks::var-tan (nks::interpret a) #:len len)]
    [(nks::tanh a len) (nks::var-tanh (nks::interpret a) #:len len)]
    [(nks::arctan a len) (nks::var-arctan (nks::interpret a) #:len len)]
    [(nks::softplus a len) (nks::var-softplus (nks::interpret a) #:len len)]
    [(nks::relu a len) (nks::var-relu (nks::interpret a) #:len len)]
    [(nks::mish a len) (nks::var-mish (nks::interpret a) #:len len)]
    [(nks::sqrt a len) (nks::var-sqrt (nks::interpret a) #:len len)]
    [(nks::rsqrt a len) (nks::var-rsqrt (nks::interpret a) #:len len)]
    [(nks::square a len) (nks::var-square (nks::interpret a))]
    [(nks::mtx::transpose a prec) (nks::mtx-transpose (nks::interpret a) #:prec prec)]
    [(nks::mtx::matmul a b rhs.T? prec) (nks::mtx-matmul (nks::interpret a) (nks::interpret b) #:rhs.T? rhs.T? #:prec prec)]
    [(nks::mtx::broadcast a axis reps) (nks::mtx-broadcast (nks::interpret a) #:axis axis #:reps reps)]
    [(nks::mtx::reduce op a axis prec) (nks::mtx-reduce op (nks::interpret a) #:axis axis #:prec prec)]
    [(nks::mtx::elemwise op a b prec) (nks::mtx-elemwise op (nks::interpret a) #:rhs (nks::interpret b) #:prec prec)]
    [(nks::mtx::activation op a b scale prec) (nks::mtx-activation op (nks::interpret a) #:bias (nks::interpret b) 
                                                                                        #:scale scale #:prec prec)]
    [(nks::mtx::rmsnorm a b prec) (nks::mtx-rmsnorm (nks::interpret a) (nks::interpret b) #:prec prec)]
    [(nks::mtx::softmax a b prec) (nks::mtx-softmax (nks::interpret a) #:bias (nks::interpret b) #:prec prec)]
    [_ (error "Illegal abstraction for NKS")]))


;; Structs to define NKI language abstractions
(struct nki::lang::add (a b prec) #:transparent #:mutable)
(struct nki::lang::subtract (a b prec) #:transparent #:mutable)
(struct nki::lang::multiply (a b prec) #:transparent #:mutable)
(struct nki::lang::divide (a b prec) #:transparent #:mutable)
(struct nki::lang::maximum (a b prec) #:transparent #:mutable)
(struct nki::lang::minimum (a b prec) #:transparent #:mutable)
(struct nki::lang::sum (a axis prec) #:transparent #:mutable)
(struct nki::lang::prod (a axis prec) #:transparent #:mutable)
(struct nki::lang::max (a axis prec) #:transparent #:mutable)
(struct nki::lang::min (a axis prec) #:transparent #:mutable)
(struct nki::lang::negative (a prec) #:transparent #:mutable)
(struct nki::lang::exp (a prec) #:transparent #:mutable)
(struct nki::lang::log (a prec) #:transparent #:mutable)
(struct nki::lang::sin (a prec) #:transparent #:mutable)
(struct nki::lang::cos (a prec) #:transparent #:mutable)
(struct nki::lang::tan (a prec) #:transparent #:mutable)
(struct nki::lang::tanh (a prec) #:transparent #:mutable)
(struct nki::lang::arctan (a prec) #:transparent #:mutable)
(struct nki::lang::sqrt (a prec) #:transparent #:mutable)
(struct nki::lang::rsqrt (a prec) #:transparent #:mutable)
(struct nki::lang::square (a prec) #:transparent #:mutable)
(struct nki::lang::relu (a prec) #:transparent #:mutable)
(struct nki::lang::mish (a prec) #:transparent #:mutable)
(struct nki::lang::softplus (a prec) #:transparent #:mutable)
(struct nki::lang::softmax (a prec) #:transparent #:mutable)
(struct nki::lang::transpose (a prec) #:transparent #:mutable)
(struct nki::lang::matmul (a b transpose_x? prec) #:transparent #:mutable)
(struct nki::lang::invert (a prec) #:transparent #:mutable)
(struct nki::lang::bitwise_and (a b prec) #:transparent #:mutable)
(struct nki::lang::bitwise_or (a b prec) #:transparent #:mutable)
(struct nki::lang::bitwise_xor (a b prec) #:transparent #:mutable)
(struct nki::lang::left_shift (a b prec) #:transparent #:mutable)
(struct nki::lang::right_shift (a b prec) #:transparent #:mutable)
(struct nki::lang::equal (a b prec) #:transparent #:mutable)
(struct nki::lang::not_equal (a b prec) #:transparent #:mutable)
(struct nki::lang::greater (a b prec) #:transparent #:mutable)
(struct nki::lang::greater_equal (a b prec) #:transparent #:mutable)
(struct nki::lang::less (a b prec) #:transparent #:mutable)
(struct nki::lang::less_equal (a b prec) #:transparent #:mutable)

;; Structs to define NKI ISA abstractions
(struct nki::isa::matmul (a b prec) #:transparent #:mutable)
(struct nki::isa::transpose (a prec) #:transparent #:mutable)
(struct nki::isa::activation (op a bias scale prec) #:transparent #:mutable)
(struct nki::isa::tensor_reduce (op a axis negate prec) #:transparent #:mutable)
(struct nki::isa::tensor_partition_reduce (op a prec) #:transparent #:mutable)
(struct nki::isa::tensor_tensor (a b op prec) #:transparent #:mutable)

;; Interpreter for NKI's APIs -- both for the language and the ISA
(define (nki::interpret program)
  (destruct program
    [(nks::var a) a]
    [(nks::mtx::broadcast a axis reps) (nks::mtx-broadcast (nki::interpret a) #:axis axis #:reps reps)]
    [(nki::lang::sum a axis prec) (nki.lang.sum (nki::interpret a) #:axis axis #:prec prec)]
    [(nki::lang::prod a axis prec) (nki.lang.prod (nki::interpret a) #:axis axis #:prec prec)]
    [(nki::lang::max a axis prec) (nki.lang.max (nki::interpret a) #:axis axis #:prec prec)]
    [(nki::lang::min a axis prec) (nki.lang.min (nki::interpret a) #:axis axis #:prec prec)]
    [(nki::lang::add a b prec) (nki.lang.add (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::subtract a b prec) (nki.lang.subtract (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::multiply a b prec) (nki.lang.multiply (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::divide a b prec) (nki.lang.divide (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::maximum a b prec) (nki.lang.maximum (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::minimum a b prec) (nki.lang.minimum (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::negative a prec) (nki.lang.negative (nki::interpret a) #:prec prec)]
    [(nki::lang::exp a prec) (nki.lang.exp (nki::interpret a) #:prec prec)]
    [(nki::lang::log a prec) (nki.lang.log (nki::interpret a) #:prec prec)]
    [(nki::lang::sin a prec) (nki.lang.sin (nki::interpret a) #:prec prec)]
    [(nki::lang::cos a prec) (nki.lang.cos (nki::interpret a) #:prec prec)]
    [(nki::lang::tan a prec) (nki.lang.tan (nki::interpret a) #:prec prec)]
    [(nki::lang::tanh a prec) (nki.lang.tanh (nki::interpret a) #:prec prec)]
    [(nki::lang::arctan a prec) (nki.lang.arctan (nki::interpret a) #:prec prec)]
    [(nki::lang::sqrt a prec) (nki.lang.sqrt (nki::interpret a) #:prec prec)]
    [(nki::lang::rsqrt a prec) (nki.lang.rsqrt (nki::interpret a) #:prec prec)]
    [(nki::lang::square a prec) (nki.lang.square (nki::interpret a) #:prec prec)]
    [(nki::lang::relu a prec) (nki.lang.relu (nki::interpret a) #:prec prec)]
    [(nki::lang::mish a prec) (nki.lang.mish (nki::interpret a) #:prec prec)]
    [(nki::lang::softplus a prec) (nki.lang.softplus (nki::interpret a) #:prec prec)]
    [(nki::lang::transpose a prec) (nki.lang.transpose (nki::interpret a) #:prec prec)]
    [(nki::lang::matmul a b transpose_x? prec) (nki.lang.matmul (nki::interpret a) (nki::interpret b) 
                                                                #:transpose_x?  transpose_x? #:prec prec)]
    [(nki::lang::softmax a prec) (nki.lang.softmax (nki::interpret a) #:prec prec)]
    [(nki::lang::invert a prec) (nki.lang.invert (nki::interpret a) #:prec prec)]
    [(nki::lang::bitwise_and a b prec) (nki.lang.bitwise_and (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::bitwise_or a b prec) (nki.lang.bitwise_or (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::bitwise_xor a b prec) (nki.lang.bitwise_xor (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::left_shift a b prec) (nki.lang.left_shift (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::right_shift a b prec) (nki.lang.right_shift (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::equal a b prec) (nki.lang.equal (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::not_equal a b prec) (nki.lang.not_equal (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::greater a b prec) (nki.lang.greater (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::greater_equal a b prec) (nki.lang.greater_equal (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::less a b prec) (nki.lang.less (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::lang::less_equal a b prec) (nki.lang.less_equal (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::isa::matmul a b prec) (nki.isa.nc_matmul (nki::interpret a) (nki::interpret b) #:prec prec)]
    [(nki::isa::transpose a prec) (nki.isa.nc_transpose (nki::interpret a) #:prec prec)]
    [(nki::isa::activation op a bias scale prec) (nki.isa.activation op (nki::interpret a) 
                                                        #:bias (nki::interpret bias) #:scale scale #:prec prec)]
    [(nki::isa::tensor_reduce op a axis negate prec) (nki.isa.tensor_reduce op (nki::interpret a) 
                                                        #:axis axis #:negate negate #:prec prec)]
    [(nki::isa::tensor_partition_reduce op a prec) (nki.isa.tensor_partition_reduce op (nki::interpret a) #:prec prec)]
    [(nki::isa::tensor_tensor a b op prec) (nki.isa.tensor_tensor (nki::interpret a) (nki::interpret b) op #:prec prec)]
    [_ (error "Illegal matrix abstraction for NKS")]))

