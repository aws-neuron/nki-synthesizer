"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Synthesizer for NKS

"""


from __future__ import annotations


import itertools
from itertools import product
import copy
import subprocess
from multiprocessing import Process, Queue
import concurrent.futures
import threading
import time
import re

from egglog import *

from ir import *
from nki_ops import *
from torch_ops import *


class Sketch:
    def __init__(self):
        self.stmts = list()
        self.grammars = list()
        self.op_to_elems = dict()
        self.op_to_shape = dict()

    def __hash__(self):
        return id(self)

    def __str__(self):
        for stmt in self.stmts:
            print(stmt)
        print("grammars:")
        print(self.grammars)
        print("op_to_elems:")
        print(self.op_to_elems)
        print("op_to_shape:")
        print(self.op_to_shape)
        return ""


# Support for equality saturation
class ExprTerm(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> ExprTerm: ...

    def __mul__(self, other: ExprTerm) -> ExprTerm: ...
    
    def __truediv__(self, other: ExprTerm) -> ExprTerm: ...
    
    

# Some hand-implemented rewrite rules for simplifying expressions.
# TODO: This needs more work; extend these rules as necessary.
a, b, c, d = vars_("a b c d", ExprTerm)

simplifying_rule_set = ruleset(
    rewrite(a * b).to(b * a), 
    rewrite(a * (b * c)).to((a * b) * c),
    rewrite((a * b) * c).to(a * (b * c)), 
    rewrite(a / (b * c)).to((a / b) / c),
    rewrite((a / b) / c).to((a / c) / b),
    rewrite(a / (b / c)).to((a / b) * c),
    rewrite((a * b) / c).to(a * (b / c)),
    rewrite(a * (b / c)).to((a / c) * b),
    #rewrite((a / b) * c).to(a * (c / b)),  
    rewrite((a / b) * b).to(a),  
    rewrite((a * b) / b).to(a),  
    rewrite((a * b) / (b * c)).to(a / c),

    #rewrite((a * b) * (c * d)).to((a * c) * (b * d)),  
    #rewrite((a * (b * (c * d)))).to(((a * b) * c) * d),  
    # rewrite((a / b) * (c / d)).to((a / d) * (c / b)), 
    # rewrite((a / b) / (c / d)).to((a / c) * (d / b)),  
    # rewrite((a * b) / (c * d)).to((a / c) * (b / d)),  
    # rewrite((a * b) * (c / d)).to((a * c) * (b / d)),
    # rewrite((a / (b / (c / d)))).to(((a / d) * (c / b))),
    rewrite(((a / b) / c) / d).to((a / c) / (b * d)),
    
    rewrite(a / a).to(ExprTerm(1)),
    rewrite(a * ExprTerm(1)).to(a),
    rewrite(a / ExprTerm(1)).to(a),
    rewrite(a * ExprTerm(0)).to(ExprTerm(0)),
    rewrite(ExprTerm(0) / a).to(ExprTerm(0)),  
)


class IndexCandidate:    
    def __init__(self):
        self.idx_pair = None
        self.shape = None
        self.loops = list()
        # Loops can be used to concatenate or accumulate tiles.
        self.loop_kinds = list()
        # EGraph for using equality saturation to 
        # simplify complex albegraic expressions.
        self.egraph = EGraph()
        
    def __eq__(self, other):
        if not isinstance(other, IndexCandidate):
            return False
        return self.idx_pair == other.idx_pair and self.shape == other.shape \
            and self.loops == other.loops and self.loop_kinds == other.loop_kinds
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((
            self.idx_pair,
            self.shape,
            tuple(self.loops),
            tuple(self.loop_kinds)
        ))

    def __str__(self):
        def stringify(value):
            return str(value) 

        shape_str = (stringify(self.shape[0]), stringify(self.shape[1])) if self.shape else None
        return f"idx_pair: {self.idx_pair}, shape: {shape_str}, loop_kinds: {self.loop_kinds}"


class NKS_Synthesizer:
    def __init__(self):
        # Function name and arguments
        self.func_name = None
        self.func_args = list()
        
        # Sketch and specification
        self.org_loopless_sketch = list()
        self.final_loopless_sketch = list()
        self.lowered_sketches = list()
        self.lifted_sketches = list()
        self.org_sketch = list()
        self.full_sketches = list()
        self.org_spec = list()

        # Track all the hole definitions after synthesis
        self.hole_defs = dict()
        
        # Search space for indexing expressions
        self.holes_to_search_space = dict()
        self.candidate_mappings = dict()
        self.holes_to_unique_candidates = list()
        
        # Rule set for equality saturation for checking equality
        self.eqcheck_rule_set = None
        
        # Track automatically generated rosette files
        # for synthesis.
        self.file_names = list()
        self.file_names_to_sketch = dict()
        
    def gen_expr(self, x):
        if isinstance(x, str):
            return ExprTerm.var(x)
        if isinstance(x, int):
            return ExprTerm(x)
        elif isinstance(x, Variable):
            return ExprTerm.var(x.name)
        return x
      
    def generate_eqcheck_rule_set(self):
      # Generate rewrite rules from the sketch
      rules_list = list()
      for op in self.org_sketch:
        if not isinstance(op, AssignmentOp):
          continue
        lhs = self.gen_expr(op.name)
        if op.op == AssignmentOp.add:
          rhs = self.gen_expr(op.operands[0]) + self.gen_expr(op.operands[1])
        elif op.op == AssignmentOp.sub:
          rhs = self.gen_expr(op.operands[0]) - self.gen_expr(op.operands[1])
        elif op.op == AssignmentOp.mul:
          rhs = self.gen_expr(op.operands[0]) * self.gen_expr(op.operands[1])
        elif op.op == AssignmentOp.div:
          rhs = self.gen_expr(op.operands[0]) / self.gen_expr(op.operands[1])
        elif op.op == AssignmentOp.none:
          rhs = self.gen_expr(op.operands[0])
        rules_list.append((lhs, rhs))
        rules_list.append((rhs, lhs))
      # Generate the rewrite rules
      add_rewrites = lambda rules: [rewrite(lhs).to(rhs) for lhs, rhs in rules]
      dynamic_rules = add_rewrites(rules_list)
      self.eqcheck_rule_set = ruleset(*dynamic_rules)
      
    def are_equal(self, expr1, expr2):
      egraph = EGraph()
      expr1 = self.gen_expr(expr1)
      expr2 = self.gen_expr(expr2)
      egraph.register(expr1)
      egraph.register(expr2)
      egraph.run(self.eqcheck_rule_set * 10)
      try:
        if egraph.check(expr1 == expr2) == None:
          return True
        return False
      except:
        return False
        
    def generate_index_search_space_for_shape(self, iterator_space, hole_op, shape):   
        for candidate in iterator_space:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 2
            i, j = candidate
            # Get the shape of the op
            search_space_candidate = IndexCandidate()
            if isinstance(i, int) and i == 0:
                if isinstance(j, int) and j == 0:
                    search_space_candidate.idx_pair = (i, j)
                    search_space_candidate.shape = (self.gen_expr(shape[0]), self.gen_expr(shape[1]))
                else:
                    assert isinstance(j, NKS_for_loop)
                    search_space_candidate.idx_pair = (i, j.iterator.name)
                    if self.are_equal(shape[1], 1) == True:
                      continue
                    expr_op = self.gen_expr(shape[1]) / self.gen_expr(j.end)
                    expr_op = search_space_candidate.egraph.simplify(expr_op, simplifying_rule_set * 10)
                    search_space_candidate.shape = (self.gen_expr(shape[0]), expr_op)
            else:
                assert isinstance(i, NKS_for_loop)
                if isinstance(j, int) and j == 0:
                    search_space_candidate.idx_pair = (i.iterator.name, j)
                    if self.are_equal(shape[0], 1) == True:
                      continue
                    expr_op = self.gen_expr(shape[0]) / self.gen_expr(i.end)
                    expr_op = search_space_candidate.egraph.simplify(expr_op, simplifying_rule_set * 10)
                    search_space_candidate.shape = (expr_op, self.gen_expr(shape[1]))
                else:
                    assert isinstance(j, NKS_for_loop)
                    search_space_candidate.idx_pair = (i.iterator.name, j.iterator.name)
                    #if shape[0] == 1 or shape[0] == "1" or shape[1] == 1 or shape[1] == "1":
                    if self.are_equal(shape[0], 1) == True or self.are_equal(shape[1], 1) == True:
                      continue
                    expr_op0 = self.gen_expr(shape[0]) / self.gen_expr(i.end)
                    expr_op0 = search_space_candidate.egraph.simplify(expr_op0, simplifying_rule_set * 10)
                    expr_op1 = self.gen_expr(shape[1]) / self.gen_expr(j.end)
                    expr_op1 = search_space_candidate.egraph.simplify(expr_op1, simplifying_rule_set * 10)
                    search_space_candidate.shape = (expr_op0, expr_op1)
            self.holes_to_search_space[hole_op].add(search_space_candidate)

    def generate_index_search_space_for_hole(self, hole_op, loop=None):
        assert isinstance(hole_op, Hole_op)        
        assert len(hole_op.operands) == 1
        operand = hole_op.operands[0]
        iterator_space = list()
        if loop is not None:
            search_space = [loop]
            if loop.parents is not None:
                found_operand = False
                for parent in loop.parents:
                    assert isinstance(parent, NKS_for_loop)
                    if isinstance(operand, TensorOp):
                        for content in parent.contents:
                            # Find cases where the return value of a loop is the operand
                            if isinstance(content, NKS_for_loop):
                                if content.ret_val == operand:
                                    found_operand = True
                                    break
                            elif content == operand:
                                found_operand = True
                                break
                        if found_operand == True:
                            break
                    search_space.append(parent)
                # Create pairs of these iterators
                iterator_space = list(itertools.combinations(search_space, 2))
            for loop_candidate in reversed(search_space):
                iterator_space.append((0, loop_candidate))
            for loop_candidate in reversed(search_space):
                iterator_space.append((loop_candidate, 0))
            #if not isinstance(operand, Tensor):
            #    iterator_space.insert(0, (0, 0))
        else:
            iterator_space.append((0, 0))
        # Generate the search space for the shape of the hole
        self.holes_to_search_space[hole_op] = set()
        if isinstance(operand, Tensor):
            # If this is a binding op, i.e., uses function arguments,
            # then we need only simple indexing expressions, and it 
            # should be easier to get the shape.
            shape = operand.shape
            assert shape is not None
            self.generate_index_search_space_for_shape(iterator_space, hole_op, shape)
        else:
            # Get the shape of the op
            for idx_candidate in self.holes_to_search_space[operand]:
                assert isinstance(idx_candidate, IndexCandidate)
                shape = idx_candidate.shape
                assert shape is not None
                self.generate_index_search_space_for_shape(iterator_space, hole_op, shape)
                
    def backtrack_candidate_mappings(self, operand):
        if isinstance(operand, TensorOp):
            # Remove candidate mappings that do not respect constraints imposed by this op
            for candidate in self.holes_to_search_space[operand].copy():
                if candidate not in self.candidate_mappings:
                  self.holes_to_search_space[operand].remove(candidate)
            if not isinstance(operand, Hole_op):
                for op_operand in operand.operands:
                    self.backtrack_candidate_mappings(op_operand)

    def generate_index_search_space_for_non_hole_op(self, op, loop=None):
        if isinstance(op, (NKS_assign, NKI_reciprocal, NKI_negative, NKI_exp, NKI_log, \
                            NKI_sin, NKI_cos, NKI_tan, NKI_tanh, NKI_arctan, NKI_sqrt, \
                            NKI_rsqrt, NKI_relu, NKI_softplus, NKI_mish, NKI_square)):
            operand = op.operands[0]
            candidates = self.holes_to_search_space[operand]
            self.holes_to_search_space[op] = set()
            for candidate in candidates:
                assert isinstance(candidate, IndexCandidate)
                search_space_candidate = IndexCandidate()
                search_space_candidate.egraph = candidate.egraph
                search_space_candidate.shape = (candidate.shape[0], candidate.shape[1])
                self.holes_to_search_space[op].add(search_space_candidate)
            return
        if isinstance(op, (NKS_transpose, NKI_nc_transpose, NKI_transpose)):
            operand = op.operands[0]
            candidates = self.holes_to_search_space[operand]
            self.holes_to_search_space[op] = set()
            for candidate in candidates:
                assert isinstance(candidate, IndexCandidate)
                search_space_candidate = IndexCandidate()
                search_space_candidate.egraph = candidate.egraph
                search_space_candidate.shape = (candidate.shape[1], candidate.shape[0])
                self.holes_to_search_space[op].add(search_space_candidate)
            return
        if isinstance(op, (NKS_reduce, NKI_max, NKI_min, NKI_sum, NKI_prod, NKI_mean)):
            operand = op.operands[0]
            axis = op.operands[1]
            candidates = self.holes_to_search_space[operand]
            self.holes_to_search_space[op] = set()
            for candidate in candidates:
                assert isinstance(candidate, IndexCandidate)
                search_space_candidate = IndexCandidate()
                search_space_candidate.egraph = candidate.egraph
                if axis == 0:
                    search_space_candidate.shape = (ExprTerm(1), candidate.shape[1])
                else:
                    assert axis == 1
                    search_space_candidate.shape = (candidate.shape[0], ExprTerm(1))
                self.holes_to_search_space[op].add(search_space_candidate)
            return
        if isinstance(op, NKS_broadcast):
            operand = op.operands[0]
            axis = op.operands[1]
            num_reps = op.operands[2]
            candidates = self.holes_to_search_space[operand]
            self.holes_to_search_space[op] = set()
            for candidate in candidates:
                assert isinstance(candidate, IndexCandidate)
                search_space_candidate = IndexCandidate()
                search_space_candidate.egraph = candidate.egraph
                if axis == 0:
                    expr_op = self.gen_expr(candidate.shape[0]) * self.gen_expr(num_reps)
                    expr_op = search_space_candidate.egraph.simplify(expr_op, simplifying_rule_set * 10)
                    search_space_candidate.shape = (expr_op, candidate.shape[1])
                else:
                    assert axis == 1
                    expr_op = self.gen_expr(candidate.shape[1]) * self.gen_expr(num_reps)
                    expr_op = search_space_candidate.egraph.simplify(expr_op, simplifying_rule_set * 10)
                    search_space_candidate.shape = (candidate.shape[0], expr_op)
                self.holes_to_search_space[op].add(search_space_candidate)
            return
        if isinstance(op, (NKI_add, NKI_subtract, NKI_multiply, NKI_divide, \
                            NKI_maximum, NKI_minimum)):
            self.holes_to_search_space[op] = set()
            operand1 = op.operands[0]
            operand2 = op.operands[1]
            # The shapes of the operands have to be the same or they should
            # be broadcastable.
            candidates1 = self.holes_to_search_space[operand1]
            candidates2 = self.holes_to_search_space[operand2]
            for candidate1 in candidates1:
                assert isinstance(candidate1, IndexCandidate)
                for candidate2 in candidates2:
                    assert isinstance(candidate2, IndexCandidate)
                    if self.are_equal(candidate1.shape[0], candidate2.shape[0]) == True \
                      and self.are_equal(candidate1.shape[1], candidate2.shape[1]) == True:
                        #if candidate1.shape == candidate2.shape:
                        if candidate1 not in self.candidate_mappings:
                            self.candidate_mappings[candidate1] = list()
                        self.candidate_mappings[candidate1].append((operand2, candidate2))
                        if candidate2 not in self.candidate_mappings:
                            self.candidate_mappings[candidate2] = list()
                        self.candidate_mappings[candidate2].append((operand1, candidate1))
                        search_space_candidate = IndexCandidate()
                        search_space_candidate.egraph = candidate1.egraph
                        search_space_candidate.shape = (candidate1.shape[0], candidate1.shape[1])
                        self.holes_to_search_space[op].add(search_space_candidate)
                    elif self.are_equal(candidate1.shape[0], candidate2.shape[0]) == True \
                      and self.are_equal(candidate1.shape[1], 1) == True:
                        if candidate1 not in self.candidate_mappings:
                            self.candidate_mappings[candidate1] = list()
                        self.candidate_mappings[candidate1].append((operand2, candidate2))
                        if candidate2 not in self.candidate_mappings:
                            self.candidate_mappings[candidate2] = list()
                        self.candidate_mappings[candidate2].append((operand1, candidate1))
                        search_space_candidate = IndexCandidate()
                        search_space_candidate.egraph = candidate2.egraph
                        search_space_candidate.shape = (candidate2.shape[0], candidate2.shape[1])
                        self.holes_to_search_space[op].add(search_space_candidate)
                    elif self.are_equal(candidate1.shape[0], candidate2.shape[0]) == True \
                      and self.are_equal(candidate2.shape[1], 1) == True:
                        if candidate1 not in self.candidate_mappings:
                            self.candidate_mappings[candidate1] = list()
                        self.candidate_mappings[candidate1].append((operand2, candidate2))
                        if candidate2 not in self.candidate_mappings:
                            self.candidate_mappings[candidate2] = list()
                        self.candidate_mappings[candidate2].append((operand1, candidate1))
                        search_space_candidate = IndexCandidate()
                        search_space_candidate.egraph = candidate1.egraph
                        search_space_candidate.shape = (candidate1.shape[0], candidate1.shape[1])
                        self.holes_to_search_space[op].add(search_space_candidate)
            # Remove candidate mappings that do not respect constraints imposed by this op
            self.backtrack_candidate_mappings(operand1)
            self.backtrack_candidate_mappings(operand2)
            return 
        if isinstance(op, NKI_activation):
            if len(op.operands) == 3:
                self.holes_to_search_space[op] = set()
                operand1 = op.operands[0]
                operand2 = op.operands[1]
                candidates1 = self.holes_to_search_space[operand1]
                candidates2 = self.holes_to_search_space[operand2]
                for candidate1 in candidates1:
                    assert isinstance(candidate1, IndexCandidate)
                    for candidate2 in candidates2:
                        assert isinstance(candidate2, IndexCandidate)
                        if self.are_equal(candidate1.shape[0], candidate2.shape[0]) == True \
                          and self.are_equal(candidate2.shape[1], 1) == True:
                            if candidate1 not in self.candidate_mappings:
                                self.candidate_mappings[candidate1] = list()
                            self.candidate_mappings[candidate1].append((operand2, candidate2))
                            if candidate2 not in self.candidate_mappings:
                                self.candidate_mappings[candidate2] = list()
                            self.candidate_mappings[candidate2].append((operand1, candidate1))
                            search_space_candidate = IndexCandidate()
                            search_space_candidate.egraph = candidate1.egraph
                            search_space_candidate.shape = (candidate1.shape[0], candidate1.shape[1])
                            self.holes_to_search_space[op].add(search_space_candidate)
                # Remove candidate mappings that do not respect constraints imposed by this op
                self.backtrack_candidate_mappings(operand1)
                self.backtrack_candidate_mappings(operand2)
            else:
                operand = op.operands[0]
                candidates = self.holes_to_search_space[operand]
                self.holes_to_search_space[op] = set()
                for candidate in candidates:
                    assert isinstance(candidate, IndexCandidate)
                    search_space_candidate = IndexCandidate()
                    search_space_candidate.egraph = candidate.egraph
                    search_space_candidate.shape = (candidate.shape[0], candidate.shape[1])
                    self.holes_to_search_space[op].add(search_space_candidate) 
            return
        if isinstance(op, NKI_nc_matmul):
            self.holes_to_search_space[op] = set()
            operand1 = op.operands[0]
            operand2 = op.operands[1]
            candidates1 = self.holes_to_search_space[operand1]
            candidates2 = self.holes_to_search_space[operand2]
            for candidate1 in candidates1:
                assert isinstance(candidate1, IndexCandidate)
                for candidate2 in candidates2:
                    assert isinstance(candidate2, IndexCandidate)
                    if self.are_equal(candidate1.shape[0], candidate2.shape[0]) == True:
                        if candidate1 not in self.candidate_mappings:
                            self.candidate_mappings[candidate1] = list()
                        self.candidate_mappings[candidate1].append((operand2, candidate2))
                        if candidate2 not in self.candidate_mappings:
                            self.candidate_mappings[candidate2] = list()
                        self.candidate_mappings[candidate2].append((operand1, candidate1))
                        search_space_candidate = IndexCandidate()
                        search_space_candidate.shape = (candidate1.shape[1], candidate2.shape[1])
                        self.holes_to_search_space[op].add(search_space_candidate)
            # Remove candidate mappings that do not respect constraints imposed by this op
            self.backtrack_candidate_mappings(operand1)
            self.backtrack_candidate_mappings(operand2)
            return   
    
    def generate_index_search_space_for_loop(self, loop):        
        for op in loop.contents:
            if isinstance(op, NKS_for_loop):
                self.generate_index_search_space_for_loop(op)
                continue
            if isinstance(op, Hole_op):
                self.generate_index_search_space_for_hole(op, loop)
                continue
            self.generate_index_search_space_for_non_hole_op(op, loop)
        # Deal with the return value of this loop:
        # this loop could accumulate, concat row-wise, or column-wise.
        op = loop.ret_val
        assert op is not None
        search_space_list = list(self.holes_to_search_space[op])
        num_candidates = len(search_space_list)
        for idx in range(num_candidates):
            candidate = search_space_list[idx]
            assert isinstance(candidate, IndexCandidate)
            if isinstance(op, NKI_nc_matmul):
                search_space_list[idx].loop_kinds.append("nks::mtx-accumulate")
                search_space_list[idx].loops.append(loop)
                # Nothing to do for accumulate. Handle other loop kinds.
                # Handle row-wise concat
                # new_candidate = IndexCandidate()
                # new_candidate.egraph = candidate.egraph
                # new_candidate.idx_pair = candidate.idx_pair
                # new_candidate.loop_kinds.append("nks::mtx-rowwise-concat")
                # expr_op = self.gen_expr(candidate.shape[0]) * self.gen_expr(loop.end)
                # expr_op = new_candidate.egraph.simplify(expr_op,  simplifying_rule_set * 10)
                # new_candidate.shape = (expr_op, candidate.shape[1])  
                # new_candidate.loops.append(loop)
                # search_space_list.append(new_candidate)
                # # # Handle column-wise concat
                # new_candidate = IndexCandidate()
                # new_candidate.egraph = candidate.egraph
                # new_candidate.idx_pair = candidate.idx_pair
                # new_candidate.loop_kinds.append("nks::mtx-colwise-concat")
                # expr_op = self.gen_expr(candidate.shape[1]) * self.gen_expr(loop.end)
                # expr_op = new_candidate.egraph.simplify(expr_op,  simplifying_rule_set * 10)
                # new_candidate.shape = (candidate.shape[0], expr_op)  
                # new_candidate.loops.append(loop)
                # search_space_list.append(new_candidate)
            else:
                # Handle row-wise concat
                search_space_list[idx].loop_kinds.append("nks::mtx-rowwise-concat")
                expr_op = self.gen_expr(candidate.shape[0]) * self.gen_expr(loop.end)
                expr_op = search_space_list[idx].egraph.simplify(expr_op, simplifying_rule_set * 10)
                search_space_list[idx].shape = (expr_op, candidate.shape[1])  
                search_space_list[idx].loops.append(loop)
                # Handle column-wise concat
                new_candidate = IndexCandidate()
                new_candidate.egraph = candidate.egraph
                new_candidate.idx_pair = candidate.idx_pair
                new_candidate.loop_kinds = candidate.loop_kinds
                new_candidate.loop_kinds.append("nks::mtx-colwise-concat")
                expr_op = self.gen_expr(candidate.shape[1]) * self.gen_expr(loop.end)
                expr_op = new_candidate.egraph.simplify(expr_op,  simplifying_rule_set * 10)
                new_candidate.shape = (candidate.shape[0], expr_op) 
                new_candidate.loops = candidate.loops
                new_candidate.loops.append(loop)
                search_space_list.append(new_candidate)  
        self.holes_to_search_space[op] = set(search_space_list)
            
    def generate_index_search_space(self):
        # Generate a search space for the index of the holes based
        # on loop nesting in the original sketch.
        for op in self.org_sketch:
            if isinstance(op, NKS_for_loop):
                self.generate_index_search_space_for_loop(op)
                continue
            if isinstance(op, Hole_op):
                self.generate_index_search_space_for_hole(op)
                continue
            self.generate_index_search_space_for_non_hole_op(op)
        # Generate multiple permutations of dictionaries that map holes
        # to indexing candidates.
        for op in list(self.holes_to_search_space.keys()).copy():
            if not isinstance(op, Hole_op):
                del self.holes_to_search_space[op]
            else:
                if len(self.holes_to_search_space[op]) == 0:
                    del self.holes_to_search_space[op]
                
        for hole_op, candidates in self.holes_to_search_space.items():
            for candidate in candidates:
                assert isinstance(candidate, IndexCandidate)
        keys = self.holes_to_search_space.keys()
        values = [list(v) for v in self.holes_to_search_space.values()]
        permutations = [dict(zip(keys, combination)) for combination in product(*values)]
        # Weed out combinations of indexing candidates that are not valid
        removed_list = list()
        for idx, mapping in enumerate(permutations):
            assert isinstance(mapping, dict)
            for hole_op, candidate in mapping.items():
                assert isinstance(hole_op, Hole_op)
                assert isinstance(candidate, IndexCandidate)
                if candidate in self.candidate_mappings:
                    match_found = False
                    for corr_op, mapped_candidates in self.candidate_mappings[candidate]:
                        if corr_op not in mapping:
                            continue
                        if mapping[corr_op] == mapped_candidates:
                            match_found = True
                            break
                    if match_found == False:
                        # Remove this mapping
                        removed_list.append(idx)
                        break
        for idx in reversed(removed_list):
            del permutations[idx]
        self.holes_to_unique_candidates = permutations
    
    def generate_full_synthesizer(self):
        # Generate the indexing search space
        self.generate_eqcheck_rule_set()
        self.generate_index_search_space()
        for hole_op, candidates in self.holes_to_search_space.items():
            if isinstance(hole_op, Hole_op):
                print("len(candidates):")
                print(len(candidates))
                for candidate in candidates:
                    assert isinstance(candidate, IndexCandidate)
                    print("\nhole_op:")
                    print(hole_op)
                    print("candidate:")
                    print(candidate)
            
    
    def emit_full_sketches(self):
        sketches = list()
        for sketch in self.full_sketches:
            sketch_str = list()
            string = "(define (sketch "
            for arg in self.func_args:
                string += "bv_" + arg.name + " "
            string += ")"
            sketch_str.append(string)
            for idx, arg in enumerate(self.func_args):
                if isinstance(arg, Tensor):
                    m, n = arg.shape
                    sketch_str.append("    (define " + arg.name + " (nks::matrix bv_" \
                                + arg.name + " " + str(m) \
                                + " " + str(n * arg.bitwidth) + "))")
            for stmt in sketch.stmts:
                sketch_str.append(stmt.to_rosette(4))
            sketch_str.append("    (nks::matrix-vect " + sketch.stmts[-1].name + ")")
            sketch_str.append(")\n\n")
            for grammar in sketch.grammars:
                sketch_str.append(grammar)
            sketches.append(sketch_str)
        return sketches
    
    def get_header(self):
        header = list()
        header.append("#lang rosette")
        header.append("(require rosette/lib/synthax)")
        header.append("(require rosette/lib/angelic)")
        header.append("(require racket/pretty)")
        header.append("(require racket/serialize)")
        header.append("(require rosette/lib/destruct)")
        header.append("(require rosette/solver/smt/z3)")
        header.append("(require \"nks_ops.rkt\")")
        header.append("(require \"nks_dsl.rkt\")")
        header.append("(require \"cegis.rkt\")")
        header.append("(require \"nki_lang.rkt\")")
        header.append("(require \"nki_isa.rkt\")")
        header.append("(require \"torch.rkt\")")
        header.append("(enable-debug)")
        return header

    def synth_invocation(self, n):
        stmt = list()
        stmt.append("(define (sketch_func params)")
        stmt.append("  (sketch ")
        for idx in range(len(self.func_args)):
            stmt.append("    (vector-ref params " + str(idx) + ")")
        stmt.append("  )")
        stmt.append(")\n\n")
        stmt.append("(define (spec_func params)")
        stmt.append("  (spec ")
        for idx in range(len(self.func_args)):
            stmt.append("    (vector-ref params " + str(idx) + ")")
        stmt.append("  )")
        stmt.append(")\n\n")
        string = "(define bitwidth-list (list "
        for arg in self.func_args:
            if isinstance(arg, Tensor):
                string += str(n * n * arg.bitwidth) + " "
        string += "))"
        stmt.append(string)
        stmt.append("(define (generate-params env)")
        string = "    (vector"
        for idx in range(len(self.func_args)):
            string += " (vector-ref env " + str(idx) + ")"
        string += ")"
        stmt.append(string)
        stmt.append(")\n\n")
        stmt.append("(define-values (satisfiable? sol? _)")
        stmt.append("   (cegis-synthesis spec_func sketch_func bitwidth-list generate-params '()))")
        stmt.append("(define t0 (current-seconds))")
        stmt.append("(displayln \"is Satisfiable?\")")
        stmt.append("(println satisfiable?)")
        stmt.append("(define t1 (current-seconds))")
        stmt.append("(- t1 t0)")
        stmt.append("(if satisfiable? '() (raise 'failed satisfiable?))")
        return stmt
    
    def verification_invocation(self):
        stmt = list()
        string = ""
        for arg in self.func_args:
            if isinstance(arg, Tensor):
                string += "(define-symbolic " + arg.name + " (bitvector "
                m, n = arg.shape
                string += str(n * (m) * arg.bitwidth) + "))\n"
        stmt.append(string)
        stmt.append("(define satisfiable?")
        stmt.append("  (verify (assert (bveq ")
        stmt.append(" (spec ")
        for arg in self.func_args:
            if isinstance(arg, Tensor):
                stmt.append("    " + arg.name)
        stmt.append(") ")
        stmt.append(" (sketch ")
        for arg in self.func_args:
            if isinstance(arg, Tensor):
                stmt.append("    " + arg.name)
        stmt.append(")))))")
        stmt.append("(displayln \"is Satisfiable?\")")
        stmt.append("(println satisfiable?)")
        stmt.append("(if satisfiable? '() (raise 'failed satisfiable?))")
        return stmt

    def get_reduction_ops_in_spec(self):
        ops = set()
        for stmt in self.org_spec:
            for op in stmt.reduction_ops():
                ops.add(op)
        return ops
    
    def gen_hole_signature(self, op, num_rows, num_cols):
        assert isinstance(op, Hole_op)
        string = "(define (" + op.opkind + " "
        args = list()
        for i in range(num_rows):
            for j in range(num_cols):
                name = "a" + str(i * num_cols + j)
                args.append(name)
                string += name + " "
        string += ")\n"
        return args, string
    
    def emit_transpose_grammar(self, op, num_rows, num_cols):
        assert isinstance(op, Hole_op)
        args, string = self.gen_hole_signature(op, num_rows, num_cols)
        string += "    (nks::interpret (choose \n"
        string += "      (nks::pack4 "
        for arg in args:
            string += "(nks::var " + arg + ") "
        string += ")\n"
        string += "      (nks::pack4 "
        for j in range(num_cols):
            for i in range(num_rows):
                string += "(nks::var " + args[i * num_cols + j] + ") "
        string += ")\n"
        string += ")))\n"
        return string

    def emit_generic_transform_grammar(self, num_pack, op, num_rows, num_cols):
        assert isinstance(op, Hole_op)
        args, hole_str = self.gen_hole_signature(op, num_rows, num_cols)
        grammar_name = op.opkind + "?"
        string = "(define-grammar (" + grammar_name + " " 
        for arg in args:
            string += arg + " "
        string += ")\n"
        if num_pack == 2:
            string += "    [expr (nks::pack2 (expr0) (expr0))]\n"
        else:
            string += "    [expr (nks::pack4 (expr0) (expr0) (expr0) (expr0))]\n"
        string += "    [expr0 (choose \n"
        for arg in args:
            string += "      (nks::var " + arg + ")\n"
        string += ")])\n"
        string += "\n\n" + hole_str
        string += "    (nks::interpret (" + grammar_name + " "
        for arg in args:
            string += arg + " " 
        string += "#:depth " + str(1) + " #:start expr)))\n"
        return string

    def create_pairs(self, args, num_rows, num_cols):
        pairs = list()
        for row in range(num_rows):
            for col in range(num_cols - 1):
                pairs.append((args[row * num_cols + col], args[row * num_cols + col + 1]))
        for col in range(num_cols):
            for row in range(num_rows - 1):
                pairs.append((args[row * num_cols + col], args[(row + 1) * num_cols + col]))
        return pairs
    
    def emit_tiered_grammar(self, args, grammar_name, num_pack, op_group, num_rows, num_cols):
        string = "(define-grammar (" + grammar_name + " " 
        for arg in args:
            string += arg + " "
        string += ")\n"
        if num_pack == 2:
            string += "    [expr (nks::pack2 (expr0) (expr0))]\n"
        else:
            string += "    [expr (nks::pack4 (expr0) (expr0) (expr0) (expr0))]\n"
        string += "    [expr0 (choose \n"
        for arg in args:
            string += "      (nks::var " + arg + ")\n"
        pairs = self.create_pairs(args, num_rows, num_cols) 
        #list(itertools.combinations(args, 2))
        for op in op_group:
            assert op == "bvadd" or op == "bvmul" or op == "bvsmax" or op == "bvsmin"
            if op == "bvadd":
                nks_op = "nks::add"
            elif op == "bvmul":
                nks_op = "nks::mul"
            elif op == "bvsmax":
                nks_op = "nks::max" 
            else:
                nks_op = "nks::min"
            for pair in pairs:
                string += "      (" + nks_op + " (nks::var " + pair[0] + ") (nks::var " + pair[1] + "))\n"
        string += ")])\n"
        return string
    
    def emit_packed_grammars(self, num_pack, num_op_groups, hole_op, num_rows, num_cols):
        assert isinstance(hole_op, Hole_op)
        # Only account for reduction ops that found in spec
        constituent_ops = self.get_reduction_ops_in_spec()
        ops = list()
        for op in constituent_ops:
            if op in {"bvadd", "bvmul", "bvsmax", "bvsmin"}:
                ops.append(op)
        holes = list()
        op_groups = list(itertools.combinations(ops, num_op_groups))
        for op_group in op_groups:
            grammar_name = hole_op.opkind + "?"
            args, hole_str = self.gen_hole_signature(hole_op, num_rows, num_cols)
            string = self.emit_tiered_grammar(args, grammar_name, num_pack, op_group, num_rows, num_cols)
            string += "\n\n" + hole_str
            string += "    (nks::interpret (" + grammar_name + " "
            for arg in args:
                string += arg + " " 
            string += "#:depth " + str(1) + " #:start expr)))\n"
            holes.append(string)
        return holes
      
    def add_stmts_to_sketches(self, stmts, sketches):
        new_sketches = list()
        for sketch in sketches:
            new_sketch = copy.deepcopy(sketch)
            new_sketch.stmts.extend(stmts)
            new_sketches.append(new_sketch)
        return new_sketches
    
    def add_grammar_to_sketches(self, grammar, stmts, sketches):
        new_sketches = list()
        for sketch in sketches:
            new_sketch = copy.deepcopy(sketch)
            new_sketch.grammars.append(grammar)
            new_sketch.stmts.extend(stmts)
            new_sketches.append(new_sketch)
        return new_sketches

    def get_output_shape(self):
        op_to_shape = dict()
        for op in self.org_spec:
            operand_shapes = list()
            for operand in op.operands:
                if isinstance(operand, Tensor):
                    operand_shapes.append(operand.shape)
                    continue
                if isinstance(operand, TensorOp):
                    op_to_shape[operand.name] = op_to_shape[operand.name]
                    operand_shapes.append(op_to_shape[operand.name])
            if isinstance(op, (Torch_add, Torch_subtract, Torch_multiply, \
                               Torch_divide, Torch_maximum, Torch_minimum)):
                assert len(operand_shapes) == 2
                if operand_shapes[1][1] != "1":
                    assert operand_shapes[0] == operand_shapes[1]
                else:
                    assert operand_shapes[0][0] == operand_shapes[1][0]
                op_to_shape[op.name] = operand_shapes[0]
                continue
            if isinstance(op, (Torch_negative, Torch_exp, Torch_log, Torch_sin, \
                               Torch_cos, Torch_tan, Torch_tanh, Torch_arctan, \
                               Torch_sqrt, Torch_rsqrt, Torch_relu, Torch_softplus, \
                               Torch_mish, Torch_square, Torch_reciprocal)):
                assert len(operand_shapes) == 1
                op_to_shape[op.name] = operand_shapes[0]
                continue
            if isinstance(op, (Torch_sum, Torch_prod, Torch_mean, Torch_min, Torch_max)):
                assert len(operand_shapes) == 1
                axis = op.operands[1]
                if axis == 0:
                    op_to_shape[op.name] = ("1", operand_shapes[0][1])
                else:
                    assert axis == 1
                    op_to_shape[op.name] = (operand_shapes[0][0], "1")
                continue
            if isinstance(op, Torch_matmul):
                assert len(operand_shapes) == 2
                m1, n1 = operand_shapes[0]
                m2, n2 = operand_shapes[1]
                assert n1 == m2
                op_to_shape[op.name] = (m1, n2)
                continue
            if isinstance(op, Torch_softmax):
                assert len(operand_shapes) == 1
                op_to_shape[op.name] = operand_shapes[0]
                continue
            if isinstance(op, Torch_transpose):
                assert len(operand_shapes) == 1
                m, n = operand_shapes[0]
                op_to_shape[op.name] = (n, m)
                continue
        return op_to_shape[self.org_spec[-1].name]

    def lower_loopless_sketch(self, num_elems, num_op_groups=1):
        sketches = [Sketch()]
        for op in self.org_loopless_sketch:
            for operand in op.operands:
                if isinstance(operand, Tensor):
                    if operand.shape != None:
                        org_m, org_n = operand.shape
                    else:
                        # Just try 16x16 for now
                        org_m, org_n = 16, 16
                    if org_m != 1 and org_n != 1:
                        operand.row_major_access(num_elems, num_elems)
                        sketches = self.add_stmts_to_sketches(operand.partial_stmts, sketches)
                        for sketch in sketches:
                            sketch.op_to_elems[operand.name] = operand.partial_elems
                            sketch.op_to_shape[operand.name] = (num_elems, num_elems)
                        continue
                    assert org_m == 1 or org_n == 1
                    operand.row_major_access(num_elems, 1)
                    sketches = self.add_stmts_to_sketches(operand.partial_stmts, sketches)
                    for sketch in sketches:
                        sketch.op_to_elems[operand.name] = operand.partial_elems
                        sketch.op_to_shape[operand.name] = (num_elems, 1)
            if isinstance(op, Hole_op):
                # Get the output size of the hole
                assert len(op.operands) == 1
                operand = op.operands[0]
                for sketch in sketches:
                    string = "(define " + op.name + " (" + op.opkind + " "
                    elems = sketch.op_to_elems[operand.name]
                    for elem in elems:
                        string += elem + " "
                    string += "))"
                    sketch.stmts.append(string)
                for sketch in sketches:
                # Account for a specialized transpose grammar
                if isinstance(operand, Tensor):
                    if operand.shape != None:
                        org_m, org_n = operand.shape
                    else:
                        # Just try 16x16 for now
                        org_m, org_n = 16, 16
                    org_sketches = sketches
                    new_sketches = list()
                    for sketch in org_sketches:
                        if org_m != 1 and org_n != 1:
                            grammar = self.emit_transpose_grammar(op, num_elems, num_elems)
                            elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                            new_sketches.append(new_sketch)
                            # Account for more generic data transforming grammars
                            # pack2_grammar = self.emit_generic_transform_grammar(2, op, num_elems, num_elems)
                            # elems, stmts = op.row_major_access(op.name, num_elems, 1)
                            # new_sketch = copy.deepcopy(sketch)
                            # new_sketch.grammars.append(pack2_grammar)
                            # new_sketch.stmts.extend(stmts)
                            # new_sketch.op_to_elems[op.name] = elems
                            # new_sketch.op_to_shape[op.name] = (num_elems, 1)
                            # new_sketches.append(new_sketch)
                            # pack4_grammar = self.emit_generic_transform_grammar(4, op, num_elems, num_elems)
                            # elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                            # new_sketch = copy.deepcopy(sketch)
                            # new_sketch.grammars.append(pack4_grammar)
                            # new_sketch.stmts.extend(stmts)
                            # new_sketch.op_to_elems[op.name] = elems
                            # new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                            # new_sketches.append(new_sketch)
                        else:
                            # Account for more generic data transforming grammars
                            assert org_m == 1 or org_n == 1
                            pack2_grammar = self.emit_generic_transform_grammar(2, op, num_elems, 1)
                            elems, stmts = op.row_major_access(op.name, num_elems, 1)
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(pack2_grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, 1)
                            new_sketches.append(new_sketch)
                            pack4_grammar = self.emit_generic_transform_grammar(4, op, num_elems, 1)
                            elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(pack4_grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                            new_sketches.append(new_sketch)
                    sketches = new_sketches
                    continue
                assert isinstance(operand, TensorOp)
                # Account multiple sketches and grammars for different ops
                org_sketches = sketches
                new_sketches = list()
                for sketch in org_sketches:
                    operand_elems = sketch.op_to_elems[operand.name]
                    if len(operand_elems) == num_elems * num_elems:
                         # Account for transpose grammar
                        transpose_grammar = self.emit_transpose_grammar(op, num_elems, num_elems)
                        elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                        new_sketch = copy.deepcopy(sketch)
                        new_sketch.grammars.append(transpose_grammar)
                        new_sketch.stmts.extend(stmts)
                        new_sketch.op_to_elems[op.name] = elems
                        new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                        new_sketches.append(new_sketch)
                        # Account for other grammars
                        pack2_grammars = self.emit_packed_grammars(2, num_op_groups, op, num_elems, num_elems)
                        elems, stmts = op.row_major_access(op.name, num_elems, 1)
                        for grammar in pack2_grammars:
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, 1)
                            new_sketches.append(new_sketch)
                        pack4_grammars = self.emit_packed_grammars(4, num_op_groups, op, num_elems, num_elems)
                        elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                        for grammar in pack4_grammars:
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                            new_sketches.append(new_sketch)
                    elif len(operand_elems) == num_elems:
                         # Account for packed grammar
                        pack2_grammars = self.emit_packed_grammars(2, num_op_groups, op, num_elems, 1)
                        elems, stmts = op.row_major_access(op.name, num_elems, 1)
                        for grammar in pack2_grammars:
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, 1)
                            new_sketches.append(new_sketch)
                        pack4_grammars = self.emit_packed_grammars(4, num_op_groups, op, num_elems, 1)
                        elems, stmts = op.row_major_access(op.name, num_elems, num_elems)
                        for grammar in pack4_grammars:
                            new_sketch = copy.deepcopy(sketch)
                            new_sketch.grammars.append(grammar)
                            new_sketch.stmts.extend(stmts)
                            new_sketch.op_to_elems[op.name] = elems
                            new_sketch.op_to_shape[op.name] = (num_elems, num_elems)
                            new_sketches.append(new_sketch)
                sketches = new_sketches
                continue
            if isinstance(op, NKI_nc_matmul):
                stationary = op.operands[0]
                moving = op.operands[1]
                remove_sketches = list()
                for sketch in sketches:
                    stationary_elems = sketch.op_to_elems[stationary.name]
                    k, m = sketch.op_to_shape[stationary.name]
                    moving_elems = sketch.op_to_elems[moving.name]
                    k1, n = sketch.op_to_shape[moving.name]
                    if k1 != k:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    op.generate_partial_ops(stationary_elems, moving_elems, m, n, k)
                    sketch.stmts.extend(op.partial_stmts)
                    sketch.op_to_elems[op.name] = op.partial_elems
                    sketch.op_to_shape[op.name] = (m, n)
                    op.partial_stmts = list()
                    op.partial_elems = list()
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, NKI_activation):
                data = op.operands[0]
                remove_sketches = list()
                for sketch in sketches:
                    data_elems = sketch.op_to_elems[data.name]
                    m, n = sketch.op_to_shape[data.name]
                    if len(op.operands) == 2:
                        op.generate_partial_ops(data_elems, m, n)
                    else:
                        bias = op.operands[1]
                        bias_elems = sketch.op_to_elems[bias.name]
                        b1, b2 = sketch.op_to_shape[bias.name]
                        if (b1 != m or b2 != 1) and (b1 != 1 or b2 != m):
                            # Remove this sketch from the list since it is not valid
                            remove_sketches.append(sketch)
                            continue
                        op.generate_partial_ops(data_elems, m, n, bias_elems)
                    sketch.stmts.extend(op.partial_stmts)
                    sketch.op_to_elems[op.name] = op.partial_elems
                    sketch.op_to_shape[op.name] = sketch.op_to_shape[data.name]
                    op.partial_stmts = list()
                    op.partial_elems = list()
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, (NKI_add, NKI_subtract, NKI_multiply, \
                                NKI_divide, NKI_maximum, NKI_minimum)):
                lhs = op.operands[0]
                rhs = op.operands[1]
                remove_sketches = list()
                for sketch in sketches:
                    lhs_elems = sketch.op_to_elems[lhs.name]
                    rhs_elems = sketch.op_to_elems[rhs.name]
                    m, n = sketch.op_to_shape[lhs.name]
                    m1, n1 = sketch.op_to_shape[rhs.name]
                    if m1 != m:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    if n != n1 and n1 != 1:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    op.generate_partial_ops(lhs_elems, m, n, rhs_elems, m1, n1)
                    sketch.stmts.extend(op.partial_stmts)
                    sketch.op_to_elems[op.name] = op.partial_elems
                    sketch.op_to_shape[op.name] = (m, n)
                    op.partial_stmts = list()
                    op.partial_elems = list()
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, (NKI_negative, NKI_exp, NKI_log, NKI_sin, NKI_cos, \
                                NKI_tan, NKI_tanh, NKI_arctan, NKI_sqrt, NKI_rsqrt, \
                                NKI_relu, NKI_softplus, NKI_mish, NKI_square, NKI_reciprocal)):
                data = op.operands[0]
                for sketch in sketches:
                    data_elems = sketch.op_to_elems[data.name]
                    m, n = sketch.op_to_shape[data.name]
                    op.generate_partial_ops(data_elems, m, n)
                    sketch.stmts.extend(op.partial_stmts)
                    sketch.op_to_elems[op.name] = op.partial_elems
                    sketch.op_to_shape[op.name] = (m, n)
                    op.partial_stmts = list()
                    op.partial_elems = list()
                continue

        # Eliminate sketches that do not have the correct output shape
        remove_sketches = list()
        org_m, org_n = self.get_output_shape()
        if org_m != 1 and org_n != 1:
            for sketch in sketches:
                sketch_output_shape = sketch.op_to_shape[self.org_loopless_sketch[-1].name]
                if sketch_output_shape[0] == 1 or sketch_output_shape[1] == 1:
                    remove_sketches.append(sketch)
        else:
            assert org_m == 1 or org_n == 1
            for sketch in sketches:
                if sketch.op_to_shape[self.org_loopless_sketch[-1].name] != (org_m, 1):
                    remove_sketches.append(sketch)
        for sketch in remove_sketches:
            sketches.remove(sketch)
        # Add a return value to the sketches
        lastop = self.org_loopless_sketch[-1]
        for sketch in sketches:
            elems = sketch.op_to_elems[lastop.name]
            string = "(concat"
            for elem in elems:
                string += " " + elem
            string += ")"
            sketch.stmts.append(string)
        self.lowered_sketches = sketches

    def emit_lowered_spec(self, n):
        spec = list()
        string = "(define (spec "
        for arg in self.func_args:
            string += "bv_" + arg.name + " "
        string += ")"
        spec.append(string)
        for arg in self.func_args:
            assert isinstance(arg, Tensor)
            spec.append("    (define " + arg.name + " (nks::matrix bv_" \
                        + arg.name + " " + str(n) + " " + str(n * arg.bitwidth) + "))")
        for stmt in self.org_spec:
            spec.append(stmt.to_rosette(4))
        spec.append("    (nks::matrix-vect " + self.org_spec[-1].name + ")")
        spec.append(")\n\n")
        return spec

    def emit_lowered_sketches(self):
        sketches = list()
        for sketch in self.lowered_sketches:
            sketch_str = list()
            string = "(define (sketch "
            for arg in self.func_args:
                string += arg.name + " "
            string += ")"
            sketch_str.append(string)
            for stmt in sketch.stmts:
                sketch_str.append(stmt)
            sketch_str.append(")\n\n")
            for grammar in sketch.grammars:
                sketch_str.append(grammar)
            sketches.append(sketch_str)
        return sketches

    def emit_lifted_spec(self):
        spec = list()
        string = "(define (spec "
        for arg in self.func_args:
            string += "bv_" + arg.name + " "
        string += ")"
        spec.append(string)
        for idx, arg in enumerate(self.func_args):
            if isinstance(arg, Tensor):
                m, n = arg.shape
                spec.append("    (define " + arg.name + " (nks::matrix bv_" \
                            + arg.name + " " + str(m) \
                            + " " + str(n * arg.bitwidth) + "))")
        for stmt in self.org_spec:
            spec.append(stmt.to_rosette(4))
        spec.append("    (nks::matrix-vect " + self.org_spec[-1].name + ")")
        spec.append(")\n\n")
        return spec
    
    def emit_lifted_sketches(self):
        sketches = list()
        for sketch in self.lifted_sketches:
            sketch_str = list()
            string = "(define (sketch "
            for arg in self.func_args:
                string += "bv_" + arg.name + " "
            string += ")"
            sketch_str.append(string)
            for idx, arg in enumerate(self.func_args):
                if isinstance(arg, Tensor):
                    m, n = arg.shape
                    sketch_str.append("    (define " + arg.name + " (nks::matrix bv_" \
                                + arg.name + " " + str(m) \
                                + " " + str(n * arg.bitwidth) + "))")
            for stmt in sketch.stmts:
                sketch_str.append(stmt.to_rosette(4))
            sketch_str.append("    (nks::matrix-vect " + sketch.stmts[-1].name + ")")
            sketch_str.append(")\n\n")
            for grammar in sketch.grammars:
                sketch_str.append(grammar)
            sketches.append(sketch_str)
        return sketches

    def generate_synthesizer(self, num_elems, num_threads=16):
        num_threads = 16
        # Fix the function argument definitions
        for idx, arg in enumerate(self.func_args):
            if isinstance(arg, Tensor):
                # Get the shape of this argument.
                for op in self.org_loopless_sketch:
                    for operand in op.operands:
                        if isinstance(operand, Tensor):
                            if operand.name == arg.name:
                                if operand.shape != None:
                                    m, n = operand.shape
                                    if isinstance(m, Variable):
                                        if m.real_val == None:
                                            m = 16
                                        else:
                                            m = m.real_val
                                    if isinstance(n, Variable):
                                        if n.real_val == None:
                                            n = 16
                                        else:
                                            n = n.real_val
                                    self.func_args[idx].shape = (m, n)
                                    break
                                else:
                                    # Just try 16x16 matrices for now
                                    self.func_args[idx].shape = (16, 16)
                                    break
        header = ""
        for line in self.get_header():
            header += line + "\n"   
        header += "\n\n"
        # Generate lowered specification
        spec_str = ""
        for line in self.emit_lowered_spec(num_elems):
            spec_str += line + "\n"
        # Generate loopless sketches
        self.lower_loopless_sketch(num_elems)
        sketches = self.emit_lowered_sketches()
        # Generate a bunch of files for synthesis
        for idx, sketch_lines in enumerate(sketches):
            string = ""
            for line in sketch_lines:
                string += line + "\n"
            string += "\n\n"
            for line in self.synth_invocation(num_elems):
                string += line + "\n"
            filename = "test_" + str(idx) + ".rkt"
            file_contents = header + spec_str + string
            f = open(filename, "w")
            f.write(file_contents)
            f.close()
            self.file_names.append(filename)
            self.file_names_to_sketch[filename] = self.lowered_sketches[idx]
        
        # Extract hole definitions from the synthesized output, and 
        # perform lifting for verification. Sometimes the synthesizer 
        # generates sketches that are hard to lift, so we need to try
        # synthesizing again, and hence the loop.
        synthesis_time = 0
        while len(self.lifted_sketches) == 0:
            # If no lifted sketches are found, then we need to try synthesizing again
            output, execution_time, file_name = self.run_synthesizer(num_threads)
            synthesis_time += execution_time
            print("synthesis_time so far:")
            print(synthesis_time)
            # Extract hole definitions from the synthesized output, and 
            # perform lifting for verification
            self.extract_synthesized_hole_defs(output)
            self.verified_lifting_of_sketch(self.file_names_to_sketch[file_name])
        print("synthesis_time:")
        print(synthesis_time)

        # Generate lifted specifications
        spec_str = ""
        for line in self.emit_lifted_spec():
            spec_str += line + "\n"
        # Generate files for lifted sketches
        sketches = self.emit_lifted_sketches()
        self.file_names = list()
        for idx, sketch_lines in enumerate(sketches):
            string = ""
            for line in sketch_lines:
                string += line + "\n"
            string += "\n\n"
            for line in self.verification_invocation():
                string += line + "\n"
            string += "\n\n"
            filename = "verify_" + str(idx) + ".rkt"
            file_contents = header + spec_str + string
            f = open(filename, "w")
            f.write(file_contents)
            f.close()
            self.file_names.append(filename)
            self.file_names_to_sketch[filename] = self.lifted_sketches[idx]

        # Run the synthesizer and get a valid sketch that works
        output, verification_time, working_file_name = self.run_synthesizer(1)
        self.final_loopless_sketch = self.file_names_to_sketch[working_file_name]
        print("verification_time: ")
        print(verification_time)
        
        # Fix the hole definitions
        self.map_hole_defs()
        # Translate the final sketch to NKI
        self.translate_nks_to_nki()

        # Print out total time
        print("\n\nsynthesis_time:")
        print(synthesis_time)
        print("verification_time:")
        print(verification_time)
        print("total_time:")
        print(synthesis_time + verification_time)
        print("\n\n")
        
    def map_hole_defs(self):
        # Fix the ambiguous hole definitions that are a list of choices
        # for verifier to verify.
        for op in self.org_loopless_sketch:
            if isinstance(op, Hole_op):
                hole_def = self.hole_defs[op]
                if isinstance(hole_def, list) and len(hole_def) != 1:
                    # This holedef needs to be redefined -- find the
                    # correct definition of this hole.
                    for oplist in hole_def:
                        for idx, final_op in enumerate(self.final_loopless_sketch.stmts):
                            if final_op.name == oplist[-1].name:
                                # Now trace back to see if this oplist
                                # matches ops in the final sketch for this hole
                                for potential_op in reversed(oplist):
                                    if final_op.name != potential_op.name:
                                        break
                                    idx -= 1
                                    final_op = self.final_loopless_sketch.stmts[idx]
                                if self.final_loopless_sketch.stmts[idx + 1].name == oplist[0].name:
                                    # Found a match!
                                    self.hole_defs[op] = oplist
                                    hole_def = [oplist]
                                    break
                        if len(hole_def) == 1:
                            break

    def run_synthesizer(self, num_threads):
        start_time = time.time()
        # Shared flag to signal threads to stop
        stop_flag = threading.Event()

        def process_file(file_name):
            if stop_flag.is_set():
                return None
            command = "racket " + file_name
            output = self.launch_processes(command, num_threads)
            if output != None:
                stop_flag.set()
            return output, file_name

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.file_names)) as executor:
            futures = {executor.submit(process_file, file_name): file_name for file_name in self.file_names}
            for future in concurrent.futures.as_completed(futures):
                output, file_name = future.result()
                if output != None:
                    # Cancel all remaining futures
                    stop_flag.set()
                    break
            executor.shutdown(wait=False)
        end_time = time.time()
        execution_time = end_time - start_time
        return output, execution_time, file_name

    def convert_nks_struct_to_nks_op(self, nks_struct):
        nks_struct_to_nks_op = {
            "nks::add": "nks::var-add", 
            "nks::mul": "nks::var-mul", 
            "nks::max": "nks::var-max", 
            "nks::min": "nks::var-min"}
        return nks_struct_to_nks_op[nks_struct]

    def get_hole_shape(self, hole_op, sketch):
        # Perform forward dataflow analysis to get some idea about the
        # shape of the output of a hole operation.
        # TODO: improve this analysis to make this more general.
        for idx, op in enumerate(self.org_loopless_sketch):
            if hole_op.name != op.name:
                continue
            for fwd_op in self.org_loopless_sketch[idx + 1:]:
                # TODO: account for other ops, although the support below is sufficient for now
                if isinstance(fwd_op, (NKI_add, NKI_subtract, NKI_multiply, \
                                        NKI_divide, NKI_maximum, NKI_minimum)):
                    lhs = fwd_op.operands[0]
                    rhs = fwd_op.operands[1]
                    if hole_op.name == lhs.name:
                        m, n = sketch.op_to_shape[rhs.name]
                        return (m, n)
                    if hole_op.name == rhs.name:
                        m, n = sketch.op_to_shape[lhs.name] 
                        return (m, n)
                    continue
                # if isinstance(fwd_op, (NKI_negative, NKI_exp, NKI_log, NKI_sin, NKI_cos, \
                #                         NKI_tan, NKI_tanh, NKI_arctan, NKI_sqrt, NKI_rsqrt, \
                #                         NKI_relu, NKI_softplus, NKI_mish, NKI_square)):
                #     data = fwd_op.operands[0]
                #     if hole_op.name == data.name:
                #         m, n = sketch.op_to_shape[data.name]
                #         return (m, n)
                #     continue
        return None, None
                        
    def verified_lifting_of_sketch(self, working_sketch):   
        # Perform verified lifting
        sketches = [Sketch()]
        for op in self.org_loopless_sketch:
            for operand in op.operands:
                if isinstance(operand, Tensor):
                    for sketch in sketches:
                        if operand.shape != None:
                            m, n = operand.shape
                            if isinstance(m, Variable):
                                if m.real_val == None:
                                    m = 16
                                else:
                                    m = m.real_val
                            if isinstance(n, Variable):
                                if n.real_val == None:
                                    n = 16
                                else:
                                    n = n.real_val
                            sketch.op_to_shape[operand.name] = (m, n)
                        else:
                            # Just try 16x16 matrices for now
                            sketch.op_to_shape[operand.name] = (16, 16)
            if isinstance(op, Hole_op):
                # Get the output size of the hole
                assert len(op.operands) == 1
                operand = op.operands[0]
                hole_def_list = self.hole_defs[op]
                assert isinstance(hole_def_list, list)
                lowered_m, lowered_n = working_sketch.op_to_shape[operand.name]
                # Detect the simplest of patterns: no transformation, transpose, reduction, broadcasting
                if self.is_row_major_pattern(hole_def_list, lowered_m, lowered_n):
                    # No transformation needed, so this hole is a pass-through
                    sythop = NKS_assign(op.name, op.operands[0])
                    self.hole_defs[op] = sythop
                    sketches = self.add_stmts_to_sketches([sythop], sketches)
                    for sketch in sketches:
                        m, n = sketch.op_to_shape[operand.name]
                        sketch.op_to_shape[sythop.name] = (m, n)
                    continue
                if self.is_transpose_pattern(hole_def_list, lowered_m, lowered_n):
                    sythop = NKS_transpose(op.name, op.operands[0])
                    self.hole_defs[op] = sythop
                    sketches = self.add_stmts_to_sketches([sythop], sketches)
                    for sketch in sketches:
                        m, n = sketch.op_to_shape[operand.name]
                        sketch.op_to_shape[sythop.name] = (n, m)
                    continue
                is_reduction, redop, axis = self.is_simple_reduction_pattern(hole_def_list, lowered_m, lowered_n)
                if is_reduction:
                    sythop = NKS_reduce(op.name, self.convert_nks_struct_to_nks_op(redop), op.operands[0], axis)
                    self.hole_defs[op] = sythop
                    sketches = self.add_stmts_to_sketches([sythop], sketches)
                    for sketch in sketches:
                        m, n = sketch.op_to_shape[operand.name]
                        if axis == 0:
                            sketch.op_to_shape[sythop.name] = (1, n)
                        else:
                            sketch.op_to_shape[sythop.name] = (m, 1)
                    continue
                is_broadcasting, axis, _ = self.is_simple_broadcast_pattern(hole_def_list, lowered_m, lowered_n)
                if is_broadcasting:
                    #sketches = self.get_hole_shape(op, sketches)
                    for sketch in sketches:
                        operand_dim = sketch.op_to_shape[operand.name][axis]
                        hole_shape = self.get_hole_shape(op, sketch)
                        if axis == 0:
                            num_reps = int(hole_shape[0] / operand_dim)
                            op_shape = (hole_shape[0], sketch.op_to_shape[operand.name][1])
                        else:
                            num_reps = int(hole_shape[1] / operand_dim)
                            op_shape = (sketch.op_to_shape[operand.name][1], hole_shape[1])
                        sythop = NKS_broadcast(op.name, operand, axis, num_reps)
                        self.hole_defs[op] = sythop
                        sketch.stmts.append(sythop)
                        sketch.op_to_shape[sythop.name] = op_shape
                    continue
                # This is a more complex pattern, we have to generate multiple sketches for verification
                is_reduction, redop = self.is_some_reduction_pattern(hole_def_list)
                is_broadcasting, _ = self.is_some_broadcasting_pattern(hole_def_list)
                assert is_reduction and is_broadcasting
                # Generate sketches for verification. We aim for depth of 4 max.
                self.hole_defs[op] = list()
                operand = op.operands[0]
                assert isinstance(operand, (Tensor, TensorOp))
                org_sketches = sketches
                new_sketches = list()
                for sketch in org_sketches:
                    m, n = sketch.op_to_shape[operand.name]
                    hole_shape = self.get_hole_shape(op, sketch)
                    if m != 1 and n != 1:
                        # reduction(axis=1), broadcast(axis=1) pattern
                        new_sketch = copy.deepcopy(sketch)  
                        op1 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), operand, axis=1)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (m, 1)
                        reps = hole_shape[1]
                        op2 = NKS_broadcast(op.name, op1, axis=1, num_reps=reps)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (m, reps)
                        self.hole_defs[op].append([op1, op2])
                        new_sketches.append(new_sketch)
                        # reduction(axis=0), broadcast(axis=0) pattern
                        new_sketch = copy.deepcopy(sketch)
                        op1 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), operand, axis=0)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (1, n)
                        reps = hole_shape[0]
                        op2 = NKS_broadcast(op.name, op1, axis=0, num_reps=reps)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (reps, n)
                        self.hole_defs[op].append([op1, op2])
                        new_sketches.append(new_sketch)
                        # Transpose, reduction(axis=1), broadcast(axis=1) pattern 
                        new_sketch = copy.deepcopy(sketch)
                        op1 = NKS_transpose(gen_name(), operand)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (n, m)
                        op2 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), op1, axis=1)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (n, 1)
                        reps = hole_shape[1]
                        op3 = NKS_broadcast(op.name, op2, axis=1, num_reps=reps)
                        new_sketch.stmts.append(op3)
                        new_sketch.op_to_shape[op3.name] = (n, reps)
                        self.hole_defs[op].append([op1, op2, op3])
                        new_sketches.append(new_sketch)
                        # Transpose, reduction(axis=0), broadcast(axis=0) pattern
                        new_sketch = copy.deepcopy(sketch)
                        op1 = NKS_transpose(gen_name(), operand)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (n, m)
                        op2 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), op1, axis=0)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (1, m)
                        reps = hole_shape[0]
                        op3 = NKS_broadcast(op.name, op2, axis=0, num_reps=reps)
                        new_sketch.stmts.append(op3)
                        new_sketch.op_to_shape[op3.name] = (reps, m)
                        self.hole_defs[op].append([op1, op2, op3])
                        new_sketches.append(new_sketch)
                        continue
                    if m == 1:
                        # Brodcast(axis=0), Transpose, reduction(axis=0) pattern
                        new_sketch = copy.deepcopy(sketch)
                        reps = hole_shape[1]
                        op1 = NKS_broadcast(gen_name(), operand, axis=0, num_reps=reps)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (reps, n)
                        op2 = NKS_transpose(gen_name(), op1)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (n, reps)
                        op3 = NKS_reduce(op.name, self.convert_nks_struct_to_nks_op(redop), op2, axis=0)
                        new_sketch.stmts.append(op3)
                        new_sketch.op_to_shape[op3.name] = (1, reps)
                        self.hole_defs[op].append([op1, op2, op3])
                        new_sketches.append(new_sketch)
                        # Brodcast(axis=0), Transpose, reduction(axis=0), broadcast(axis=0) pattern
                        new_sketch = copy.deepcopy(sketch)
                        reps1 = hole_shape[1]
                        reps2 = hole_shape[0]
                        op1 = NKS_broadcast(gen_name(), operand, axis=0, num_reps=reps1)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (reps1, n)
                        op2 = NKS_transpose(gen_name(), op1)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (n, reps1)
                        op3 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), op2, axis=0)
                        new_sketch.stmts.append(op3)
                        new_sketch.op_to_shape[op3.name] = (1, reps1)
                        op4 = NKS_broadcast(op.name, op3, axis=0, num_reps=reps2)
                        new_sketch.stmts.append(op4)
                        new_sketch.op_to_shape[op4.name] = (reps2, reps1)
                        self.hole_defs[op].append([op1, op2, op3, op4])
                        new_sketches.append(new_sketch)
                        # Brodcast(axis=0), reduction(axis=0) pattern
                        new_sketch = copy.deepcopy(sketch)
                        reps = hole_shape[1]
                        op1 = NKS_broadcast(gen_name(), operand, axis=0, num_reps=reps)
                        new_sketch.stmts.append(op1)
                        new_sketch.op_to_shape[op1.name] = (reps, n)
                        op2 = NKS_reduce(op.name, self.convert_nks_struct_to_nks_op(redop), op1, axis=0)
                        new_sketch.stmts.append(op2)
                        new_sketch.op_to_shape[op2.name] = (1, reps)
                        self.hole_defs[op].append([op1, op2])
                        new_sketches.append(new_sketch)
                        continue
                    assert n == 1
                    # Brodcast(axis=1), Transpose, reduction(axis=1) pattern
                    new_sketch = copy.deepcopy(sketch)
                    reps = hole_shape[0]
                    op1 = NKS_broadcast(gen_name(), operand, axis=1, num_reps=reps)
                    new_sketch.stmts.append(op1)
                    new_sketch.op_to_shape[op1.name] = (n, reps)
                    op2 = NKS_transpose(gen_name(), op1)
                    new_sketch.stmts.append(op2)
                    new_sketch.op_to_shape[op2.name] = (reps, n)
                    op3 = NKS_reduce(op.name, self.convert_nks_struct_to_nks_op(redop), op2, axis=1)
                    new_sketch.stmts.append(op3)
                    new_sketch.op_to_shape[op3.name] = (reps, 1)
                    self.hole_defs[op].append([op1, op2, op3])
                    new_sketches.append(new_sketch)
                    # Brodcast(axis=1), Transpose, reduction(axis=1), broadcast(axis=1) pattern
                    new_sketch = copy.deepcopy(sketch)
                    reps1 = hole_shape[0]
                    reps2 = hole_shape[1]
                    op1 = NKS_broadcast(gen_name(), operand, axis=1, num_reps=reps1)
                    new_sketch.stmts.append(op1)
                    new_sketch.op_to_shape[op1.name] = (n, reps1)
                    op2 = NKS_transpose(gen_name(), op1)
                    new_sketch.stmts.append(op2)
                    new_sketch.op_to_shape[op2.name] = (reps1, n)
                    op3 = NKS_reduce(gen_name(), self.convert_nks_struct_to_nks_op(redop), op2, axis=1)
                    new_sketch.stmts.append(op3)
                    new_sketch.op_to_shape[op3.name] = (reps1, 1)
                    op4 = NKS_broadcast(op.name, op3, axis=1, reps=reps2)
                    new_sketch.stmts.append(op4)
                    new_sketch.op_to_shape[op4.name] = (reps1, reps2)
                    self.hole_defs[op].append([op1, op2, op3, op4])
                    new_sketches.append(new_sketch)
                    # Brodcast(axis=1), reduction(axis=1) pattern
                    new_sketch = copy.deepcopy(sketch)
                    reps = hole_shape[0]
                    op1 = NKS_broadcast(gen_name(), operand, axis=1, num_reps=reps)
                    new_sketch.stmts.append(op1)
                    new_sketch.op_to_shape[op1.name] = (n, reps)
                    op2 = NKS_reduce(op.name, self.convert_nks_struct_to_nks_op(redop), op1, axis=1)
                    new_sketch.stmts.append(op2)
                    new_sketch.op_to_shape[op2.name] = (reps, 1)
                    self.hole_defs[op].append([op1, op2])
                    new_sketches.append(new_sketch)
                sketches = new_sketches
                continue
            if isinstance(op, NKI_nc_matmul):
                stationary = op.operands[0]
                moving = op.operands[1]
                remove_sketches = list()
                for sketch in sketches:
                    k, m = sketch.op_to_shape[stationary.name]
                    k1, n = sketch.op_to_shape[moving.name]
                    if k != k1:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    sketch.op_to_shape[op.name] = (m, n)
                    sketch.stmts.append(op)
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, (NKI_activation)):
                data = op.operands[0]
                remove_sketches = list()
                for sketch in sketches:
                    m, n = sketch.op_to_shape[data.name]
                    if len(op.operands) == 3:
                        bias = op.operands[1]
                        b1, b2 = sketch.op_to_shape[bias.name]
                        if (b1 != m or b2 != 1) and (b1 != 1 or b2 != m):
                            # Remove this sketch from the list since it is not valid
                            remove_sketches.append(sketch)
                            continue
                    sketch.op_to_shape[op.name] = (m, n)
                    sketch.stmts.append(op)
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, (NKI_add, NKI_subtract, NKI_multiply, \
                                NKI_divide, NKI_maximum, NKI_minimum)):
                lhs = op.operands[0]
                rhs = op.operands[1]
                remove_sketches = list()
                for sketch in sketches:
                    m, n = sketch.op_to_shape[lhs.name]
                    m1, n1 = sketch.op_to_shape[rhs.name]
                    if m1 != m:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    if n != n1 and n1 != 1:
                        # Remove this sketch from the list since it is not valid
                        remove_sketches.append(sketch)
                        continue
                    sketch.op_to_shape[op.name] = (m, n)
                    sketch.stmts.append(op)
                for sketch in remove_sketches:
                    sketches.remove(sketch)
                continue
            if isinstance(op, (NKI_negative, NKI_exp, NKI_log, NKI_sin, NKI_cos, \
                                NKI_tan, NKI_tanh, NKI_arctan, NKI_sqrt, NKI_rsqrt, \
                                NKI_relu, NKI_softplus, NKI_mish, NKI_square, NKI_reciprocal)):
                data = op.operands[0]
                for sketch in sketches:
                    m, n = sketch.op_to_shape[data.name]
                    sketch.op_to_shape[op.name] = (m, n)
                    sketch.stmts.append(op)
                continue
        self.lifted_sketches = sketches

    def is_row_major_pattern(self, hole_def, num_rows, num_cols):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")

        if not all(isinstance(elem, list) and elem[0] == "nks::var" for elem in hole_def[1:]):
            return False

        num_vars = len(hole_def[1:])
        if num_vars != num_rows * num_cols:
            return False

        row_major_vars = [f"a{row * num_cols + col}" \
                    for row in range(num_rows) for col in range(num_cols)]
        hole_vars = [elem[1] for elem in hole_def[1:]]
        return hole_vars == row_major_vars

    def is_transpose_pattern(self, hole_def, num_rows, num_cols):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")

        if not all(isinstance(elem, list) \
            and elem[0] == "nks::var" for elem in hole_def[1:]):
            return False
        
        num_vars = len(hole_def[1:])
        if num_vars != num_rows * num_cols:
            return False

        transposed_vars = [f"a{(i % num_cols) * num_rows + (i // num_cols)}" for i in range(num_vars)]
        row_major_vars = [elem[1] for elem in hole_def[1:]]
        return row_major_vars == transposed_vars

    def is_simple_reduction_pattern(self, hole_def, num_rows, num_cols):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")
        # Flatten the list of operands to get the variables
        reduction_ops = {"nks::max", "nks::min", "nks::add", "nks::mul"}
        vars_list = list()
        found_redop = None
        for operand in hole_def[1:]:
            if isinstance(operand, list) and operand[0] in reduction_ops:
                if found_redop == None:
                    found_redop = operand[0]
                if found_redop != operand[0]:
                    return False, None, None
                vars_list.extend(operand[1:])
            else:
                return False, None, None
        
        if len(vars_list) != num_rows * num_cols:
            return False, None, None
        
        # Check reduction along rows
        reduction_along_cols = True
        for row in range(num_rows):
            row_vars = [f"a{row * num_cols + col}" for col in range(num_cols)]
            if not all(isinstance(elem, list) and elem[0] == "nks::var" and elem[1] \
                in row_vars for elem in vars_list[row * num_cols:(row + 1) * num_cols]):
                reduction_along_cols = False
        # Check reduction along columns
        reduction_along_rows = True
        for col in range(num_cols):
            col_vars = [f"a{row * num_cols + col}" for row in range(num_rows)]
            if not all(isinstance(elem, list) and elem[0] == "nks::var" and elem[1] \
                in col_vars for elem in vars_list[col::num_cols]):
                reduction_along_rows = False
        if reduction_along_cols:
            return True, found_redop, 1
        if reduction_along_rows:
            return True, found_redop, 0
        return False, None, None

    def is_simple_broadcast_pattern(self, hole_def, num_rows, num_cols):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")
        # Flatten the list of operands to get the variables
        vars_list = list()
        for operand in hole_def[1:]:
            if isinstance(operand, list) and operand[0] == "nks::var":
                vars_list.append(operand)
            else:
                return False, None, 0
        if len(vars_list) % (num_rows * num_cols) != 0:
            return False, None, 0
        repetitions = len(vars_list) // (num_rows * num_cols)
        # Check broadcasting pattern along rows
        broadcasting_along_rows = True
        for row in range(num_rows):
            row_vars = [f"a{row * num_cols + col}" for col in range(num_cols)]
            if not all(isinstance(elem, list) and elem[0] == "nks::var" and elem[1] == row_vars[0] \
                for elem in vars_list[row * num_cols * repetitions:(row + 1) * num_cols * repetitions]):
                broadcasting_along_rows = False
        # Check broadcasting pattern along columns
        broadcasting_along_columns = True
        for col in range(num_cols):
            col_vars = [f"a{row * num_cols + col}" for row in range(num_rows)]
            if not all(isinstance(elem, list) and elem[0] == "nks::var" and elem[1] == col_vars[0] \
                for elem in vars_list[col::num_cols * repetitions]):
                broadcasting_along_columns = False
        if broadcasting_along_rows:
            return True, 0, repetitions
        if broadcasting_along_columns:
            return True, 1, repetitions
        return False, None, 0

    def is_some_reduction_pattern(self, hole_def):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")
        # Checks for a reduction pattern by verifying the presence of 
        # reduction operations and their operands. 
        reduction_ops = {"nks::max", "nks::min", "nks::add", "nks::mul"}
        found_redop = None
        for operand in hole_def[1:]:
            if isinstance(operand, list) and operand[0] in reduction_ops:
                if found_redop == None:
                    found_redop = operand[0]
                if found_redop != operand[0]:
                    return False, None
                for var in operand[1:]:
                    if not (isinstance(var, list) and var[0] == "nks::var"):
                        return False, None
                continue
            return False, None
        return True, found_redop

    def is_some_broadcasting_pattern(self, hole_def):
        assert isinstance(hole_def, list)
        assert hole_def[0].startswith("nks::pack")
        # It also checks for broadcasting patterns by detecting if any 
        # variable is repeated in the list of operand.
        reduction_ops = {"nks::max", "nks::min", "nks::add", "nks::mul"}
        vars_list = list()
        for operand in hole_def[1:]:
            if isinstance(operand, list) and operand[0] == "nks::var":
                vars_list.append(operand[1])
                continue
            if isinstance(operand, list) and operand[0] in reduction_ops:
                for var in operand[1:]:
                    if isinstance(var, list) and var[0] == "nks::var":
                        vars_list.append(var[1])
                        continue
                    return False, 0
                continue
            return False, 0
        vars_set = set(vars_list)
        if len(vars_set) < len(vars_list):
            # Estimate the number of repetitions
            repetitions = len(vars_list) // len(vars_set)
            return True, repetitions
        return False, 0

    def extract_synthesized_hole_defs(self, input_string):
        for op in self.org_loopless_sketch:
            if isinstance(op, Hole_op):
                pattern = re.compile(r'\(define \(' + re.escape(op.opkind) \
                            + r' [^)]+\)\s*\(nks::interpret[\s\S]*?\)\)\)')
                matches = pattern.findall(input_string)
                assert matches != None
                self.hole_defs[op] = self.get_nested_lists(matches[-1])

    def get_nested_lists(self, synthesized_hole):
        # Remove the define part and split by spaces to get tokens
        synthesized_hole = synthesized_hole.split('\n', 2)[2].strip()
        tokens = synthesized_hole.replace('(', ' ( ').replace(')', ' ) ').split()
        stack = list()
        current_list = list()
        for token in tokens:
            if token == '(':
                new_list = list()
                if stack:
                    stack[-1].append(new_list)
                stack.append(new_list)
                current_list = new_list
            elif token == ')':
                if len(stack) > 1:
                    stack.pop()
                current_list = stack[-1]
            else:
                current_list.append(token)
        return stack[0]
    
    def run_command(self, command):
        try:
            sub_process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sub_process.communicate()
        except Exception as e:
            return "", "Exception {}".format(command)

    def worker(self, command, id, done_queue, print_queue):
        try:
            sub_process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = sub_process.communicate()
            stdout = stdout.decode("utf-8", "ignore")
            stderr = stderr.decode("utf-8", "ignore")
        except Exception as e:
            print_queue.put(f"Process {id}: Error: {str(e)}\n")
        finally:
            done_queue.put((id, stdout, stderr))

    def launch_processes(self, command, num_processes):
        done_queue = Queue()
        print_queue = Queue()
        processes = list()
        for i in range(num_processes):
            p = Process(target=self.worker, args=(command, i, done_queue, print_queue))
            processes.append(p)
            p.start()
        
        # Thread to print messages
        def print_worker():
            while True:
                msg = print_queue.get()
                if msg is None:
                    break
                print(msg, end='')

        print_thread = Process(target=print_worker)
        print_thread.start()

        # Wait for one of the processes to finish
        finished_process_id, stdout, stderr = done_queue.get()
        if stderr == "":
            print(f"Process {finished_process_id} finished first. Terminating all processes.\n")
            # Terminate all processes
            for p in processes:
                p.terminate()
                p.join()
            # Signal the printer thread to stop
            print_queue.put(None)
            print_thread.join()
            # Kill all z3 and racket processes
            self.run_command("killall z3")
            self.run_command("killall racket")
            return stdout
        return None

    def print(self):
        print("Original Spec:")
        for spec in self.org_spec:
            print(spec)
        print("Loopless Sketch:")
        for op in self.org_loopless_sketch:
            print(op)
        print("Original Full Sketch:")
        for op in self.org_sketch:
            print(op)
