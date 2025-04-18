;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains the semantics of general matrix operations.
;; The semantics of NKI language and ISA operations are based on
;; the semantics defined here.
;;
;;;;


#lang rosette

(require "bvops.rkt")


(provide 
    nks::var-add nks::var-sub nks::var-mul nks::var-div nks::var-mod 
    nks::var-rem nks::var-max nks::var-min nks::var-eq nks::var-gt
    nks::var-ge nks::var-lt nks::var-le nks::var-and nks::var-or nks::var-xor 
    nks::var-not nks::var-neg nks::var-square nks::var-exp nks::var-log 
    nks::var-softplus nks::var-sin nks::var-cos nks::var-tan nks::var-tanh 
    nks::var-arctan nks::var-sqrt nks::var-rsqrt nks::var-mish nks::var-relu 
    nks::var-reciprocal nks::var-shl nks::var-lshr nks::var-ashr nks::var-zero? 
    nks::var-pack nks::var-unpack nks::var-extract nks::extract nks::strided-extract 
    nks::strided-concat nks::accumulate nks::var-constant
    nks::matrix nks::matrix-vect nks::matrix-rows nks::matrix-cols 
    nks::mtx-extract nks::mtx-strided-extract nks::mtx-colwise-concat nks::mtx-rowwise-concat 
    nks::mtx-accumulate nks::mtx-transpose nks::mtx-matmul nks::mtx-elemwise nks::mtx-reduce 
    nks::mtx-window-reduce nks::mtx-scan nks::mtx-activation nks::mtx-rmsnorm 
    nks::mtx-softmax nks::mtx-broadcast nks::mtx-print nks::range)


;; Provides range of indices for loops indexing into Rosette bitvectors.
(define-syntax nks::range
  (syntax-rules ()
    ((_ h) (reverse (range 0 h 1)))
    ((_ h s) (reverse (range 0 h s)))
    ((_ l h s) (reverse (range l h s)))))


(define (nks::var-extract x idx #:len len #:prec prec)
    (define low (- (- len 1) (+ idx prec -1)))
    (define high (- (- len 1) idx))
    (extract high low x))


(define (nks::extract x idx #:len len)
    (extract (+ idx len -1) idx x))


(define (nks::strided-extract x idx #:stride stride #:len len #:reps reps)
    (apply concat
        (for/list ([i (nks::range reps)])
            (define low (+ idx (* i stride)))
            (define high (+ low len -1))
            (extract high low x))))


(define (nks::strided-concat xs #:stride stride #:len len)
    (apply concat 
        (for*/list ([i (nks::range len stride)][x xs])
                (extract (+ i stride -1) i x))))


(define (nks::accumulate xs #:len len #:prec prec)
    (apply concat 
        (for/list ([i (nks::range len prec)])
            (apply bvadd
                (for/list ([x xs])
                        (extract (+ i prec -1) i x))))))


(define nks::var-pack
  (lambda xs
    (apply concat xs)))


(define nks::var-unpack
  (lambda (xs #:len len #:prec prec)
    (apply values 
        (for/list ([i (nks::range len prec)])
            (extract (+ i prec -1) i xs)))))


(define (nks::var-constant val #:len len)
    (bv val len))

(define (nks::scalar? x)
    (bv? x))

(define nks::var-add
  (lambda xs
    (apply bvadd xs)))

(define nks::var-mul
  (lambda xs
    (apply bvmul xs)))

(define nks::var-max
  (lambda xs
    (apply bvsmax xs)))

(define nks::var-min
  (lambda xs
    (apply bvsmin xs)))

(define (nks::var-sub x y)
    (bvsub x y))

(define (nks::var-div x y)
    (bvsdiv x y))

(define (nks::var-mod x y)
    (bvsmod x y))

(define (nks::var-rem x y)
    (bvsrem x y))

(define (nks::var-eq x y)
    (bveq x y))

(define (nks::var-gt x y)
    (bvsgt x y))

(define (nks::var-ge x y)
    (bvsge x y))

(define (nks::var-lt x y)
    (bvslt x y))

(define (nks::var-le x y)
    (bvsle x y))

(define (nks::var-and x y)
    (bvand x y))

(define (nks::var-or x y)
    (bvor x y))

(define (nks::var-xor x y)
    (bvxor x y))

(define (nks::var-not x)
    (bvnot x))

(define (nks::var-neg x)
    (bvneg x))

(define (nks::var-zero? x)
    (bvzero? x))

(define (nks::var-shl x y)
    (bvshl x y))

(define (nks::var-lshr x y)
    (bvlshr x y))

(define (nks::var-ashr x y)
    (bvashr x y))

(define (nks::var-square x)
    (bvmul x x))

(define (nks::var-exp x #:len len)
    (bvexp x #:len len))

(define (nks::var-log x #:len len)
    (bvlog x #:len len))

(define (nks::var-softplus x #:len len)
    (bvsoftplus x #:len len))

(define (nks::var-sin x #:len len)
    (bvsin x #:len len))

(define (nks::var-cos x #:len len)
    (bvcos x #:len len))

(define (nks::var-tan x #:len len)
    (bvtan x #:len len))

(define (nks::var-tanh x #:len len)
    (bvtanh x #:len len))

(define (nks::var-arctan x #:len len)
    (bvarctan x #:len len))

(define (nks::var-sqrt x #:len len)
    (bvsqrt x #:len len))

(define (nks::var-rsqrt x #:len len)
    (bvrsqrt x #:len len))

(define (nks::var-mish x #:len len)
    (bvmish x #:len len))

(define (nks::var-relu x #:len len)
    (bvrelu x #:len len))

(define (nks::var-reciprocal x #:len len)
    (bvsdiv (nks::var-constant 1 #:len len) x))


(define custom-ops (list nks::var-exp nks::var-log nks::var-softplus nks::var-mish nks::var-relu
                            nks::var-sin nks::var-cos nks::var-tan nks::var-tanh nks::var-arctan
                            nks::var-sqrt nks::var-rsqrt nks::var-reciprocal))

(define (custom-op? op)
    (ormap (lambda (item) (equal? op item)) custom-ops))


;; Struct for record keeping of shapes of vectors
;; during synthesis and verfication.
(struct nks::matrix (vect rows cols) #:transparent)


(define (nks::mtx-extract mtx #:row-idx row-idx #:col-idx col-idx #:rows rows #:cols cols)
    (define mtx-rows (nks::matrix-rows mtx))
    (define mtx-cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (define bv-idx (+ (* row-idx rows mtx-cols) (* col-idx cols)))
    (define resbv (nks::strided-extract vect bv-idx #:stride mtx-cols #:len cols #:reps rows))
    (nks::matrix resbv rows cols))


(define (nks::mtx-strided-extract mtx #:row-idx row-idx #:col-idx col-idx 
                                    #:row-stride row-stride #:col-stride col-stride 
                                    #:row-len row-len #:col-len col-len #:rows rows #:cols cols)
    (define mtx-rows (nks::matrix-rows mtx))
    (define mtx-cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (define result
        (apply concat
        (for*/list ([i (nks::range row-idx (+ row-idx (* row-stride row-len)) row-stride)]
                    [j (nks::range col-idx (+ col-idx (* col-stride col-len)) col-stride)])
            (define low (+ (* i mtx-cols) j))
            (define high (+ low cols -1))
            (extract high low vect))))
    (nks::matrix result rows cols))


(define nks::mtx-colwise-concat
    (lambda xs
        (begin
            (define rows (nks::matrix-rows (list-ref xs 0)))
            (define cols (nks::matrix-cols (list-ref xs 0)))
            (define result 
                (apply concat 
                    (for*/list ([i (nks::range (* rows cols) cols)][x xs])
                        (extract (+ i cols -1) i (nks::matrix-vect x)))))
            (nks::matrix result rows (* cols (length xs))))))


(define nks::mtx-rowwise-concat
    (lambda xs
        (begin
            (define rows (nks::matrix-rows (list-ref xs 0)))
            (define cols (nks::matrix-cols (list-ref xs 0)))
            (define result 
                (apply concat 
                    (for/list ([x xs])
                        (nks::matrix-vect x))))
            (nks::matrix result (* rows (length xs)) cols))))


(define (nks::mtx-accumulate xs #:prec prec)
    (define rows (nks::matrix-rows (list-ref xs 0)))
    (define cols (nks::matrix-cols (list-ref xs 0)))
    (define result
        (apply concat 
            (for/list ([i (nks::range (* rows cols) prec)])
                (apply nks::var-add
                    (for/list ([x xs])
                        (extract (+ i prec -1) i (nks::matrix-vect x)))))))
    (nks::matrix result rows cols))


(define (nks::mtx-transpose mtx #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (define result
        (apply concat 
            (for*/list ([i (nks::range cols prec)][j (nks::range rows)])
                (define low (+ i (* j cols)))
                (define high (+ low prec -1))
                (extract high low vect))))
    (nks::matrix result (/ cols prec) (* rows prec)))


(define (nks::mtx-matmul lhs rhs #:lhs.T? [lhs.T? #f] #:prec prec)
    (define rows_a (nks::matrix-rows lhs))
    (define cols_a (nks::matrix-cols lhs))
    (define vect_a (nks::matrix-vect lhs))
    (define cols_b (nks::matrix-cols rhs))
    (define vect_b (nks::matrix-vect rhs))
    (if lhs.T?
        (begin
            (define result
                (apply concat 
                    (for*/list ([i (nks::range cols_a prec)][j (nks::range cols_b prec)])
                        (apply nks::var-add
                            (for/list ([p (nks::range rows_a)])
                                (define low_a (+ i (* p cols_a)))
                                (define high_a (+ low_a prec -1))
                                (define a (extract high_a low_a vect_a))
                                (define low_b (+ j (* p cols_b)))
                                (define high_b (+ low_b prec -1))
                                (define b (extract high_b low_b vect_b))
                                (nks::var-mul a b))))))
            (nks::matrix result (/ cols_a prec) cols_b))
        ;; Case where matrix A is not transposed
        (begin
            (define result
                (apply concat 
                    (for*/list ([i (nks::range rows_a)][j (nks::range cols_b prec)])
                        (apply nks::var-add
                            (for/list ([k (nks::range cols_a prec)])
                                (define low_a (+ k (* i cols_a)))
                                (define high_a (+ low_a prec -1))
                                (define a (extract high_a low_a vect_a))
                                (define low_b (+ j (* (/ k prec) cols_b)))
                                (define high_b (+ low_b prec -1))
                                (define b (extract high_b low_b vect_b))
                                (nks::var-mul a b))))))
            (nks::matrix result rows_a cols_b))))


(define (nks::mtx-broadcast mtx #:axis axis #:reps reps)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (if (equal? axis 0)
        (begin 
            (define result
                (apply concat
                    (for/list ([j (nks::range reps)])
                        vect)))
            (nks::matrix result (* rows reps) cols))
        ;; Account for when axis is 1
        (begin 
            (define result
                (apply concat
                    (for/list ([i (nks::range rows)])
                        (define low (* i cols))
                        (define high (+ low cols -1))
                        (define col (extract high low vect))
                        (apply concat
                            (for/list ([j (nks::range reps)])
                                col)))))
            (nks::matrix result rows (* cols reps)))))


(define (nks::mtx-activation op mtx #:bias [bias #f] #:scale [scale #f] #:prec prec)
  (define rows (nks::matrix-rows mtx))
  (define cols (nks::matrix-cols mtx))
  (define vect (nks::matrix-vect mtx))
  (define result
    (if bias
        (let ([bias-vect (nks::matrix-vect bias)])
            (if scale
                (cond
                    [(custom-op? op)
                        (apply concat
                            (for/list ([i (nks::range rows)])
                                (define bias-low (* i prec))
                                (define bias-high (+ bias-low prec -1))
                                (define ext-bias (extract bias-high bias-low bias-vect))
                                (apply concat
                                    (for/list ([j (nks::range cols prec)])
                                        (define low (+ (* i cols) j))
                                        (define high (+ low prec -1))

                                        (op (nks::var-add (nks::var-mul 
                                                (extract high low vect) scale) ext-bias) #:len prec)))))]
                    [else
                        (apply concat
                            (for/list ([i (nks::range rows)])
                                (define bias-low (* i prec))
                                (define bias-high (+ bias-low prec -1))
                                (define ext-bias (extract bias-high bias-low bias-vect))
                                (apply concat
                                    (for/list ([j (nks::range cols prec)])
                                        (define low (+ (* i cols) j))
                                        (define high (+ low prec -1))
                                        (op (nks::var-add (nks::var-mul (extract high low vect) scale) ext-bias))))))])
                ;; Case without scaling
                (cond
                    [(custom-op? op)
                        (apply concat
                            (for/list ([i (nks::range rows)])
                                (define bias-low (* i prec))
                                (define bias-high (+ bias-low prec -1))
                                (define ext-bias (extract bias-high bias-low bias-vect))
                                (apply concat
                                    (for/list ([j (nks::range cols prec)])
                                        (define low (+ (* i cols) j))
                                        (define high (+ low prec -1))
                                        (op (nks::var-add (extract high low vect) ext-bias) #:len prec)))))]
                    [else
                        (apply concat
                            (for/list ([i (nks::range rows)])
                                (define bias-low (* i prec))
                                (define bias-high (+ bias-low prec -1))
                                (define ext-bias (extract bias-high bias-low bias-vect))
                                (apply concat
                                    (for/list ([j (nks::range cols prec)])
                                        (define low (+ (* i cols) j))
                                        (define high (+ low prec -1))
                                        (op (nks::var-add (extract high low vect) ext-bias))))))])))
        ;; There is no bias
        (if scale
            (cond
                [(custom-op? op)
                    (apply concat
                        (for/list ([i (nks::range rows)])
                            (apply concat
                                (for/list ([j (nks::range cols prec)])
                                    (define low (+ (* i cols) j))
                                    (define high (+ low prec -1))
                                    (op (nks::var-mul (extract high low vect) scale) #:len prec)))))]
                [else
                    (apply concat
                        (for/list ([i (nks::range rows)])
                            (apply concat
                                (for/list ([j (nks::range cols prec)])
                                    (define low (+ (* i cols) j))
                                    (define high (+ low prec -1))
                                    (op (nks::var-mul (extract high low vect) scale))))))])
            ;; Case without scaling and no bias
            (cond
                [(custom-op? op)
                    (apply concat
                        (for/list ([i (nks::range rows)])
                            (apply concat
                                (for/list ([j (nks::range cols prec)])
                                    (define low (+ (* i cols) j))
                                    (define high (+ low prec -1))
                                    (op (extract high low vect) #:len prec)))))]
                [else
                    (apply concat
                        (for/list ([i (nks::range rows)])
                            (apply concat
                                (for/list ([j (nks::range cols prec)])
                                    (define low (+ (* i cols) j))
                                    (define high (+ low prec -1))
                                    (op (extract high low vect))))))]))))
    (nks::matrix result rows cols))


(define (nks::mtx-window-reduce op mtx #:window window #:strides strides #:shape shape #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (define row-stride (car strides))
    (define col-stride (cdr strides))
    (define win-row (car window))
    (define win-col (cdr window)) 
    (define shape-rows (car shape))
    (define shape-cols (cdr shape))
    (define pooled-rows (+ (/ (- shape-rows win-row) row-stride) 1))
    (define pooled-cols (+ (/ (- shape-cols win-col) col-stride) 1))
    (define shape-size (* shape-rows shape-cols))
    (define result
        (apply concat
            (for*/list ([p (nks::range rows)][q (nks::range (/ cols shape-size))])
                (define low (+ (* p cols) (* q shape-size)))
                (define high (+ low shape-size -1))
                (define ext-vect (extract high low vect))
                (apply concat
                    (for*/list ([i (nks::range pooled-rows)][j (nks::range pooled-cols)])
                        (apply op
                            (for*/list ([m (nks::range win-row)][n (nks::range win-col prec)])
                                (define low (+ (* (+ (* i row-stride) m) shape-cols) (+ (* j col-stride) n)))
                                (define high (+ low prec -1))
                                (extract high low ext-vect))))))))
    (nks::matrix result rows (* (/ cols shape-size) pooled-rows pooled-cols prec)))


(define (nks::mtx-reduce op mtx #:axis axis #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (if (equal? axis 0)
        (begin
            (define result
                (apply concat
                    (for/list ([j (nks::range cols prec)])
                        (apply op
                            (for/list ([i (nks::range rows)])
                                (define low (+ (* i cols) j))
                                (define high (+ low prec -1))
                                (extract high low vect))))))
            (nks::matrix result 1 cols))
        ;; Reduction along axis 1            
        (begin
            (define result
                (apply concat
                    (for/list ([i (nks::range rows)])
                        (apply op
                            (for/list ([j (nks::range cols prec)])
                                (define low (+ (* i cols) j))
                                (define high (+ low prec -1))
                                (extract high low vect))))))
            (nks::matrix result rows prec))))


(define (nks::mtx-scan op0 op1 data0 data1 #:init initial #:reverse0 [reverse0 #f] #:reverse1 [reverse1 #f] #:prec prec)
    (define rows (nks::matrix-rows data0))
    (define cols (nks::matrix-cols data0))
    (define data0_vect (nks::matrix-vect data0))
    (define data1_vect (nks::matrix-vect data1))
    ;; Check if initial is a "scalar" bitvector
    (if (bv? initial)
        (begin
            (define result
                (apply concat
                    (for/list ([i (nks::range rows)])
                        (define prev_col (bitvector->integer initial))
                        (apply concat
                            (for/list ([j (nks::range cols prec)])
                                (define low (+ (* i cols) j))
                                (define high (+ low prec -1))
                                (define elem1 (extract high low data0_vect))
                                (define elem2 (extract high low data1_vect))
                                (define prev_vect (integer->bitvector prev_col (bitvector prec)))
                                (define intres (if reverse0 (op0 prev_vect elem1) (op0 elem1 prev_vect)))
                                (define res_col (if reverse1 (op1 elem2 intres) (op1 intres elem2)))
                                (set! prev_col (bitvector->integer res_col))
                                res_col)))))
                (nks::matrix result rows cols))
        ;; initial is matrix
        (begin
            (define result
                (apply concat
                    (for/list ([i (nks::range rows)])
                        (define low (* i cols))
                        (define high (+ low prec -1))
                        (define prev_col (bitvector->integer (extract high low initial)))
                        (apply concat
                            (for/list ([j (nks::range cols prec)])
                                (define low (+ (* i cols) j))
                                (define high (+ low prec -1))
                                (define elem1 (extract high low data0_vect))
                                (define elem2 (extract high low data1_vect))
                                (define prev_vect (integer->bitvector prev_col (bitvector prec)))
                                (define intres (if reverse0 (op0 prev_vect elem1) (op0 elem1 prev_vect)))
                                (define res_col (if reverse1 (op1 elem2 intres) (op1 intres elem2)))
                                (set! prev_col (bitvector->integer res_col))
                                res_col)))))
                (nks::matrix result rows cols))))


(define (nks::mtx-rmsnorm mtx weight #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (define wtvec (nks::matrix-vect weight))
    (define result
        (apply concat
            (for/list ([i (nks::range rows)])
                (define square_sum 
                    (apply nks::var-add 
                        (for/list ([j (nks::range cols prec)])
                            (define low (+ (* i cols) j))
                            (define high (+ low prec -1))
                            (define elem (extract high low vect))
                            (nks::var-mul elem elem))))
                (define mean (nks::var-div square_sum (nks::var-constant (/ cols prec) #:len prec)))
                (define rms_reciprocal (nks::var-rsqrt mean #:len prec))
                (apply concat
                    (for/list ([j (nks::range cols prec)])
                        (define low (+ (* i cols) j))
                        (define high (+ low prec -1))
                        (nks::var-mul (extract high low vect) rms_reciprocal (extract (+ j prec -1) j wtvec)))))))
    (nks::matrix result rows cols))


(define (nks::mtx-softmax mtx #:axis axis #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (if (equal? axis 1)
      (begin
          (define result
              (apply concat
                  (for/list ([i (nks::range rows)])
                      (define max-val 
                          (apply nks::var-max 
                              (for/list ([j (nks::range cols prec)])
                                  (define low (+ (* i cols) j))
                                  (define high (+ low prec -1))
                                  (extract high low vect))))
                      (define exp-elems 
                          (apply concat
                              (for/list ([j (nks::range cols prec)])
                                  (define low (+ (* i cols) j))
                                  (define high (+ low prec -1))
                                  (nks::var-exp (nks::var-sub (extract high low vect) max-val) #:len prec))))
                      (define sum-exp
                          (apply nks::var-add
                              (for/list ([j (nks::range cols prec)])
                                  (extract (+ j prec -1) j exp-elems))))
                      (define sum-exp-reciprocal (nks::var-reciprocal sum-exp #:len prec))
                      (apply concat
                          (for/list ([j (nks::range cols prec)])
                              (nks::var-mul (extract (+ j prec -1) j exp-elems) sum-exp-reciprocal))))))
          (nks::matrix result rows cols))
      ;; Axis is 0
      (begin 
          (define result
              (apply concat
                  (for/list ([j (nks::range cols prec)])
                      (define max-val 
                          (apply nks::var-max
                              (for/list ([i (nks::range rows)])
                                  (define low (+ (* i cols) j))
                                  (define high (+ low prec -1))
                                  (extract high low vect))))
                      (define exp-elems 
                          (apply concat
                              (for/list ([i (nks::range rows)])
                                  (define low (+ (* i cols) j))
                                  (define high (+ low prec -1))
                                  (nks::var-exp (nks::var-sub (extract high low vect) max-val) #:len prec))))
                      (define sum-exp
                          (apply nks::var-add
                              (for/list ([i (nks::range (* rows prec) prec)])
                                  (extract (+ i prec -1) i exp-elems))))
                      (define sum-exp-reciprocal (nks::var-reciprocal sum-exp #:len prec))
                      (apply concat
                          (for/list ([i (nks::range (* rows prec) prec)])
                              (nks::var-mul (extract (+ i prec -1) i exp-elems) sum-exp-reciprocal))))))
          (define res.t (nks::matrix result (/ cols prec) (* rows prec)))
          (nks::mtx-transpose res.t #:prec prec))))
      

;; The rhs can be a matrix, vector or a scalar; if it is a vector, it is broadcast
;; along the dimension 1 of the lhs matrix, and if it is a scalar, it is broadcast along
;; dimensions 0 and 1 of the lhs matrix. 
;; Likewise, the lhs can be a matrix, vector or a scalar, with similar broadcasting semantics.
(define (nks::mtx-elemwise op lhs #:rhs [rhs #f] #:prec prec)
    (if rhs
        (if (bv? rhs)
            (if (bv? lhs)
                (op lhs rhs)
                (let ([lhs_cols (nks::matrix-cols lhs)]
                      [lhs_rows (nks::matrix-rows lhs)]
                      [lhs_vect (nks::matrix-vect lhs)])
                    (define result
                        (apply concat 
                            (for/list ([i (nks::range (* lhs_rows lhs_cols) prec)])
                                (op (extract (+ i prec -1) i lhs_vect) rhs))))
                    (nks::matrix result lhs_rows lhs_cols)))
            (if (bv? lhs)
                (let ([rhs_cols (nks::matrix-cols rhs)]
                      [rhs_rows (nks::matrix-rows rhs)]
                      [rhs_vect (nks::matrix-vect rhs)])
                    (define result
                        (apply concat 
                            (for/list ([i (nks::range (* rhs_rows rhs_cols) prec)])
                                (op lhs (extract (+ i prec -1) i rhs_vect)))))
                        (nks::matrix result rhs_rows rhs_cols))
                (let ([rhs_cols (nks::matrix-cols rhs)]
                      [rhs_rows (nks::matrix-rows rhs)]
                      [rhs_vect (nks::matrix-vect rhs)]
                      [lhs_cols (nks::matrix-cols lhs)]
                      [lhs_rows (nks::matrix-rows lhs)]
                      [lhs_vect (nks::matrix-vect lhs)])
                    (if (equal? lhs_cols rhs_cols)
                        (begin
                            (define result
                                (apply concat 
                                    (for/list ([i (nks::range (* lhs_rows lhs_cols) prec)])
                                        (op (extract (+ i prec -1) i lhs_vect) 
                                            (extract (+ i prec -1) i rhs_vect)))))
                            (nks::matrix result lhs_rows lhs_cols))
                        (if (equal? rhs_cols prec)
                            (begin 
                                (define result
                                    (apply concat 
                                        (for*/list ([i (nks::range lhs_rows)][j (nks::range lhs_cols prec)])
                                                (define lhs_low (+ (* i lhs_cols) j))
                                                (define lhs_high (+ lhs_low prec -1))
                                                (define rhs_low (* i prec))
                                                (define rhs_high (+ rhs_low prec -1))
                                                (op (extract lhs_high lhs_low lhs_vect) 
                                                    (extract rhs_high rhs_low rhs_vect)))))
                                (nks::matrix result lhs_rows lhs_cols))
                            (if (equal? lhs_cols prec)
                                (begin 
                                    (define result
                                        (apply concat 
                                            (for*/list ([i (nks::range rhs_rows)][j (nks::range rhs_cols prec)])
                                                    (define rhs_low (+ (* i rhs_cols) j))
                                                    (define rhs_high (+ rhs_low prec -1))
                                                    (define lhs_low (* i prec))
                                                    (define lhs_high (+ lhs_low prec -1))
                                                    (op (extract lhs_high lhs_low lhs_vect)
                                                        (extract rhs_high rhs_low rhs_vect)))))
                                    (nks::matrix result rhs_rows rhs_cols))
                                    (assert #f "Invalid operation: lhs and rhs have incompatible shapes.")))))))
        ;; Case when rhs is a matrix
        (if (custom-op? op)
            (let ([cols (nks::matrix-cols lhs)]
                  [rows (nks::matrix-rows lhs)]
                  [vect (nks::matrix-vect lhs)])
                (define result
                    (apply concat 
                        (for/list ([i (nks::range (* rows cols) prec)])
                            (op (extract (+ i prec -1) i vect) #:len prec))))
                (nks::matrix result rows cols))
            (let ([cols (nks::matrix-cols lhs)]
                  [rows (nks::matrix-rows lhs)]
                  [vect (nks::matrix-vect lhs)])
                (define result
                    (apply concat 
                        (for/list ([i (nks::range (* rows cols) prec)])
                            (op (extract (+ i prec -1) i vect)))))
                (nks::matrix result rows cols)))))
                    

(define (nks::mtx-print mtx #:name name #:prec prec)
    (define rows (nks::matrix-rows mtx))
    (define cols (nks::matrix-cols mtx))
    (define vect (nks::matrix-vect mtx))
    (printf name)
    (printf ":\n")
    (for/list ([i (nks::range rows)])
        (printf "[ ")
        (for/list ([j (nks::range cols prec)])
            (define low (+ j (* i cols)))
            (define high (+ low prec -1))
            (print (extract high low vect))
            (printf ", "))
        (printf "]\n")))