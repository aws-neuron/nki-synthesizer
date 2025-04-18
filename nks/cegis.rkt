;;;;
;; 
;; Copyright (c) 2025, Amazon.com. All Rights Reserved
;;
;; This file contains a CEGIS implementation for syntax-guided synthesis.
;;
;;;;



#lang rosette

(require rosette/lib/synthax)
(require rosette/lib/angelic)
(require racket/pretty)
(require racket/serialize)
(require racket/sandbox)
(require data/bit-vector)
(require (only-in racket build-vector))
(require (only-in racket build-list))


(provide cegis-synthesis enable-debug disable-debug)


;; Define the current bitwidth and memory limit
(current-bitwidth 16)
(custodian-limit-memory (current-custodian) (* 20000 1024 1024))


;; Debugging interfaces
(define debug-switch 0)

(define (enable-debug)
  (set! debug-switch 1))

(define (disable-debug)
  (set! debug-switch 0))

(define (debug-log message)
  (if (equal? debug-switch 1) 
    (if (struct? message) 
      (pretty-print message) 
      (displayln message))
    ;; debug-switch is 0
    '()))


(define (make-sym-bvs bwlist)
  (define (helper i)
    (define-symbolic* sym_bv (bitvector (list-ref bwlist i)))
    sym_bv)
  (build-vector (length bwlist) helper))


(define (make-conc-bv bw)
    (if (<= bw 16)
        (begin
            (define max-val (expt 2 bw))
            (define rand-val (random max-val))
            (integer->bitvector rand-val (bitvector bw)))
        (concat (make-conc-bv (/ bw 2)) (make-conc-bv (/ bw 2)))))


(define (make-conc-bvs bwlist)
    (define (helper i)
        (make-conc-bv (list-ref bwlist i)))
    (build-vector (length bwlist) helper))


(define (get-concrete-asserts assert-query-fn cex-ls)
    (define (helper i)
        (assert-query-fn (list-ref cex-ls i)))
    (build-list (length cex-ls) helper))


(define (verify-solution sketch spec bwlist assert-query-fn params)
    (define start (current-seconds))
    (debug-log "Attempting to verify synthesized solution")
    (define symbols (make-sym-bvs bwlist))
    ;; Get any counterexamples that may exist
    (define cex (verify (assert-query-fn symbols)))
    (define end (current-seconds))
    (debug-log (format "Verification took ~a seconds\n" (- end start)))
    (debug-log cex)
    (begin
        (if (sat? cex)
            ;; Counterexamples found
            (begin
            (debug-log "Verification failed :(")
            (define (helper i)
                (evaluate (vector-ref symbols i) cex))
            (define new-bvs (build-vector (length bwlist) helper))
            (debug-log new-bvs)
            (define spec_res (spec (params new-bvs)))
            (debug-log spec_res)
            (define synth_res (sketch (params new-bvs)))
            (debug-log
            (format "Verification failed ...\n\tspec produced: ~a\n\tsynthesized result produced: ~a\n"
                    spec_res
                    synth_res))
            (values #f new-bvs))
            ;; No counter examples exist!
            (values #t '()) )))


(define (cegis-synthesis spec sketch bwlist params cexs)
    ;; If the cexs is empty create a random set of concrete inputs;
    ;; othwise, we use the concrete inputs accumulated so far.
    (define cex-ls (if (equal? (length cexs) 0) (list (make-conc-bvs bwlist)) cexs))
    (debug-log "Concrete counter examples:")
    (debug-log cex-ls)
    (define (assert-query-fn env)
        (assert (bveq (sketch (params env)) (spec (params env)))))
    ;; Perform synthsis on random inputs.
    (define start_time (current-seconds))
    (debug-log "*** concrete-synthesis ***")
    (define sol? (solve (get-concrete-asserts assert-query-fn cex-ls)))
    (debug-log "Concrete synthesis done!")
    (define end_time (current-seconds))
    (define elapsed_time (- end_time start_time))
    (define satisfiable? (sat? sol?))
    (debug-log satisfiable?)
    (define tempsol (if satisfiable? (evaluate sketch sol?) '()))
    (if satisfiable? 
        ;; Verify if the current solution is correct for ALL inputs
        (begin
            (debug-log "Unchecked solution:")
            (debug-log tempsol)
            (print-forms sol?)
            (debug-log "Testing intermediate solution")
            (debug-log (spec (params (list-ref cex-ls 0))))
            (debug-log (evaluate (tempsol (params (list-ref cex-ls 0))) sol?))
            (define (exec-synth-sol env)
            (evaluate (tempsol env) sol?))
            (define (assert-query-mat-fn env)
            (define parameters-f1 (params env))
            (define parameters-f2 (params env))
            (assert (equal? (exec-synth-sol parameters-f1) (spec parameters-f2))))
            (define-values (verified? new-cex)
            (verify-solution exec-synth-sol spec bwlist assert-query-mat-fn params))
            (if verified? 
                ;; If solution is found to be correct for all possible inputs
                (values satisfiable? tempsol elapsed_time)
                ;; If not verified then attempt synthesizing with appended counter example
                (cegis-synthesis sketch spec bwlist params (append cex-ls (list new-cex)))))
        (values satisfiable? tempsol elapsed_time)))
 