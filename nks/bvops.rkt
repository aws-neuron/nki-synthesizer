;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; Custom bitvector ops currently missing in Rosette.
;;
;;;;


#lang rosette

(provide bvexp bvlog bvsoftplus bvmish bvrelu
        bvsin bvcos bvtan bvtanh bvarctan bvsqrt bvrsqrt)


(define (bvexp x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (bvadd (bvadd (bvadd (bv 1 len) x)
                (bvsdiv (bvmul x x) (bv 2 len)))
                (bvsdiv (bvmul x x x) (bv 6 len))))


(define (bvlog x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (define x1 (bvsub x (bv 1 len)))
  (define x2 (bvmul x1 x1))
  (define x3 (bvmul x2 x1))
  (bvsub (bvsub x1 (bvsdiv x2 (bv 2 len))) (bvsdiv x3 (bv 3 len))))


(define (bvsoftplus x #:len len)
  (bvlog (bvadd (bv 1 len) (bvexp x len))))


(define (bvrelu x #:len len)
    (bvsmax x (bv 0 len)))


(define (bvsin x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (define x2 (bvmul x x))
  (define x3 (bvmul x2 x))
  (define x5 (bvmul x3 x2))
  (define x7 (bvmul x5 x2))
  (bvsub (bvsub (bvsub x (bvsdiv x3 (bv 6 len))) 
                (bvsdiv x5 (bv 120 len))) (bvsdiv x7 (bv 5040 len))))


(define (bvcos x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (define x2 (bvmul x x))
  (define x4 (bvmul x2 x2))
  (define x6 (bvmul x4 x2))
  (bvsub (bvsub (bvsub (bv 1 len) (bvsdiv x2 (bv 2 len))) 
                (bvsdiv x4 (bv 24 len))) (bvsdiv x6 (bv 720 len))))


(define (bvtan x #:len len)
  (bvsdiv (bvsin x #:len len) (bvcos x #:len len)))


(define (bvtanh x #:len len)
  (define ex (bvexp x #:len len))
  (define negex (bvexp (bvneg x) #:len len))
  (bvsdiv (bvsub ex negex) (bvadd ex negex)))


(define (bvarctan x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (define x2 (bvmul x x))
  (define x3 (bvmul x2 x))
  (define x5 (bvmul x3 x2))
  (define x7 (bvmul x5 x2))
  (bvsub (bvsub (bvsub x (bvsdiv x3 (bv 3 len))) 
                (bvsdiv x5 (bv 5 len))) (bvsdiv x7 (bv 7 len))))


(define (bvmish x #:len len)
  (bvmul x (bvtan (bvsoftplus x len) len)))


(define (bvsqrt x #:len len)
  ;; Approximation using Taylor series for synthesis purposes
  (define t (bvsub x (bv 1 len)))
  (define t2 (bvmul t t))
  (define t3 (bvmul t t2))
  (define t4 (bvmul t t3))
  (bvsub (bvadd (bvsub (bvadd (bv 1 len) (bvsdiv t (bv 2 len))) 
          (bvsdiv t2 (bv 8 len))) (bvsdiv t3 (bv 16 len)))
          (bvsdiv (bvmul (bv 5 len) t4) (bv 128 len))))


;; Model using sqrt(x) / x
(define (bvrsqrt x #:len len)
  (define t (bvsub x (bv 1 len)))
  (define t2 (bvmul t t))
  (define t3 (bvmul t t2))
  (define t4 (bvmul t t3))
  (bvsub (bvadd (bvsub (bvadd (bvsdiv (bv 1 len) x) (bvsdiv t (bvmul (bv 2 len) x))) 
          (bvsdiv t2 (bvmul (bv 8 len) x))) (bvsdiv t3 (bvmul (bv 16 len) x)))
          (bvsdiv (bvmul (bv 5 len) t4) (bvmul (bv 128 len) x))))
 