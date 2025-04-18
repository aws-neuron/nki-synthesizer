"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

NKI code generator.

"""

import ast
import os

from synthgen import *



# Support for translating NKS calls to Python
class AssignmentTransformer(ast.NodeTransformer):
    def __init__(self, synthgen : SynthesizerGenerator):
        assert isinstance(synthgen, SynthesizerGenerator)
        super().__init__()
        self.counter = 0
        self.synthgen = synthgen

    def get_elemenwise_op(self, op):
        if op == "nks::var-max":
            return "max"
        if op == "nks::var-min":
            return "min"
        if op == "nks::var-div":
            return "div"
        if op == "nks::var-mod":
            return "mod"
        if op == "nks::var-rem":
            return "rem"
        return op[2:]
    
    def visit_Assign(self, node):
        # Handle cases where hole operations and NKI operations are used
        if not isinstance(node.value, ast.Call):
            return node
        if not isinstance(node.value.func, ast.Name):
            return node
        if node.value.func.id != "hole_op":
            return node
        for hole_op, hole_def in self.synthgen.hole_defs.items():
            if node != hole_op.ast_expr:
                continue
            if isinstance(hole_def, TensorOp):
                # Check if this hole must be a load + hole_def
                operand = hole_op.operands[0]
                if isinstance(operand, Tensor):
                    # Create two new function calls for nl.load + hole_def
                    new_func_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='nl', ctx=ast.Load()),
                            attr='load',
                            ctx=ast.Load()
                        ),
                        args=node.value.args,
                        keywords=[]
                    )
                    if isinstance(hole_def, NKS_assign):
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return node
                    if isinstance(hole_def, NKS_transpose):
                        temp_assign = ast.Assign(
                            targets=[ast.Name(id=f"temp_{self.counter}", ctx=ast.Store())],
                            value=new_func_call
                        )
                        temp_assign.lineno = node.lineno
                        temp_assign.col_offset = node.col_offset
                        # Create a new function call node for nks.transpose
                        new_func_call2 = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='transpose',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=f"temp_{self.counter}", ctx=ast.Load())],
                            keywords=[]
                        )
                        self.counter += 1
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call2
                        return [temp_assign, node]
                    if isinstance(hole_def, NKS_reduce):
                        temp_assign = ast.Assign(
                            targets=[ast.Name(id=f"temp_{self.counter}", ctx=ast.Store())],
                            value=new_func_call
                        )
                        temp_assign.lineno = node.lineno
                        temp_assign.col_offset = node.col_offset
                        self.counter += 1
                        # Create a new function call node for nks.reduce
                        new_func_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='reduce',
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.Constant(value=self.get_elemenwise_op(hole_def.elemwise_ops[0])),
                                ast.Name(id=f"temp_{self.counter}", ctx=ast.Load())
                            ],
                            keywords=[
                                ast.keyword(arg='axis', value=ast.Constant(value=hole_def.operands[1]))
                            ]
                        )
                        self.counter += 1
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return [temp_assign, node]
                    if isinstance(hole_def, NKS_broadcast):
                        temp_assign = ast.Assign(
                            targets=[ast.Name(id=f"temp_{self.counter}", ctx=ast.Store())],
                            value=new_func_call
                        )
                        temp_assign.lineno = node.lineno
                        temp_assign.col_offset = node.col_offset
                        # Create a new function call node for nks.broadcast
                        new_func_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='broadcast',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=f"temp_{self.counter}", ctx=ast.Load())],
                            keywords=[
                                ast.keyword(arg='axis', value=ast.Constant(value=hole_def.operands[1])),
                                ast.keyword(arg='num_reps', value=ast.Constant(value=hole_def.operands[2]))
                            ]
                        )
                        self.counter += 1
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return [temp_assign, node]
                else:
                    assert isinstance(operand, TensorOp)
                    if isinstance(hole_def, NKS_assign):
                        # Replace the right-hand side with the new function call
                        node.value = ast.Name(id=operand.name, ctx=ast.Load())
                        return node
                    if isinstance(hole_def, NKS_transpose):
                        # Create a new function call node for nks.transpose
                        new_func_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='transpose',
                                ctx=ast.Load()
                            ),
                            args=node.value.args,
                            keywords=[]
                        )     
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return node
                    if isinstance(hole_def, NKS_reduce):
                        # Create a new function call node for nks.reduce
                        new_func_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='reduce',
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.Constant(value=self.get_elemenwise_op(hole_def.elemwise_ops[0])),
                                node.value.args[0]
                            ],
                            keywords=[
                                ast.keyword(arg='axis', value=ast.Constant(value=hole_def.operands[1]))
                            ]
                        )
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return node
                    if isinstance(hole_def, NKS_broadcast):
                        # Create a new function call node for nks.broadcast
                        new_func_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='nks', ctx=ast.Load()),
                                attr='broadcast',
                                ctx=ast.Load()
                            ),
                            args=node.value.args,
                            keywords=[
                                ast.keyword(arg='axis', value=ast.Constant(value=hole_def.operands[1])),
                                ast.keyword(arg='num_reps', value=ast.Constant(value=hole_def.operands[2]))
                            ]
                        )
                        # Replace the right-hand side with the new function call
                        node.value = new_func_call
                        return node
                    

class NKICodegen:
    def __init__(self, SynthGen, ast_tree):
        transformer = AssignmentTransformer(SynthGen)
        new_tree = transformer.visit(ast_tree)
        new_source_code = ast.unparse(new_tree)
        headers = "\n\nimport neuronxcc.nki.language as nl"
        headers += "\nimport neuronxcc.nki.isa as nisa"
        headers += "\nimport neuronxcc.nki.compiler as ncc"
        headers += "\nimport math"
        headers += "\nimport numpy as np"
        headers += "\nfrom neuronxcc import nki"
        headers += "\nfrom neuronxcc.nki.language import par_dim"
        new_source_code = headers + "\n\n\n" + new_source_code
        new_source_code = new_source_code.replace("sketch", "nki.jit")
        with open("generated_nki_kernel.py", "w") as f:
            f.write(new_source_code)
        print(new_source_code)
    