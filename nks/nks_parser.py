"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Generator of code for specification and sketch for NKS.

"""


import ast

from ir import *
from nki_ops import *
from torch_ops import *


class BaseSemaGenerator(ast.NodeVisitor):
    calls_to_ignore = {"tuple", "affine_range", "sequential_range", 
                       "ndarray", "par_dim", "zeros", "ceil"}

    def __init__(self, ast_tree):        
        # General semantic information
        self.func_name = None
        self.func_args = None
        self.return_val = None

        # Mapping between tensor and variable names to their definitions
        self.val_defs = dict()

        # Tracks all assignment operations
        self.assignments = list()
        
        # Track name to variable mapping
        self.name_to_var = dict()
        
        # Track loop bounds which can be tweaked as needed
        self.loop_bounds = list()

        # Function calls that have been accounted for and are safe to skip.
        self.skip_calls = set()

        self.replacement_vals = dict()
        
        # Track parsed statements
        self.statements = list()

        # Visit every node of AST tree
        self.visit(ast_tree)

    def _get_lhs(self, node_target):
        if isinstance(node_target, ast.Subscript):
            assert isinstance(node_target.value, ast.Name)
            lhs = node_target.value.id 
        else:
            assert isinstance(node_target, ast.Name)
            lhs = node_target.id 
        # Make sure that the lhs is not a redefintion
        if lhs in self.val_defs:
            i = 0
            while lhs + str(i) in self.val_defs:
                i += 1
            self.replacement_vals[lhs] = lhs + str(i)
            lhs = self.replacement_vals[lhs]
        return lhs

    def _parse_keywords(self, keywords, arg_name):
        for keyword in keywords:
            assert isinstance(keyword, ast.keyword)
            assert keyword.arg == "tile_shape"
            assert isinstance(keyword.value, ast.List)
            shape_dims = list()
            for key_arg in keyword.value.elts:
                if isinstance(key_arg, ast.Name):
                    # If there is a real value associated with this key argument
                    # is known, then get that value.
                    if key_arg.id in self.val_defs:
                        shape_dims.append(self.val_defs[key_arg.id])
                    else:
                        shape_dims.append(Variable(key_arg.id))
                else:
                    assert isinstance(key_arg, ast.Constant)
                    shape_dims.append(key_arg.value)
            self.val_defs[arg_name] = Tensor(arg_name, shape_dims)
    
    def _parse_arg(self, arg, keywords=None):
        if isinstance(arg, ast.Name):
            operand = arg.id
            if operand in self.replacement_vals:
                assert self.replacement_vals[operand] in self.val_defs
                if keywords:
                    self._parse_keywords(keywords, self.replacement_vals[operand])
                return self.val_defs[self.replacement_vals[operand]]
            else:
                if keywords:
                    self._parse_keywords(keywords, arg.id)
                return self.val_defs[arg.id]
        elif isinstance(arg, ast.Subscript):
            assert isinstance(arg.value, ast.Name)
            operand = arg.value.id
            if operand in self.replacement_vals:
                assert self.replacement_vals[operand] in self.val_defs
                if keywords:
                    self._parse_keywords(keywords, self.replacement_vals[operand])
                return self.val_defs[self.replacement_vals[operand]]
            else:
                if keywords:
                    self._parse_keywords(keywords, arg.value.id)
                return self.val_defs[arg.value.id]
        else: 
            assert isinstance(arg, ast.Constant)
            return arg.value

    def visit_Module(self, node):
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.func_name = node.name
        self.func_args = [arg.arg for arg in node.args.args]
        for idx, arg in enumerate(self.func_args):
            self.val_defs[arg] = Tensor(arg)
        self.generic_visit(node)

    def visit_Return(self, node):
        assert isinstance(node.value, ast.Name)
        self.return_val = node.value.id
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Assignment of constant values
        if isinstance(node.value, ast.Constant):
            if type(node.value.value) == float:
                assert isinstance(node.targets[0], ast.Name)
                self.val_defs[node.targets[0].id] = Variable(node.targets[0].id, node.value.value)
                var = Variable("", node.value.value)
                op = AssignmentOp(node.targets[0].id, AssignmentOp.none, [var], node)
                self.assignments.append(op)
                self.func_args.append(self.val_defs[node.targets[0].id])
            self.generic_visit(node)
            return
        # Handle cases where one variable is assigned to another
        if isinstance(node.value, ast.Name):
            assert isinstance(node.targets[0], ast.Name)
            if node.value.id not in self.name_to_var:
                var = Variable(node.value.id)
                self.name_to_var[node.value.id] = var
            else:
                var = self.name_to_var[node.value.id]
            op = AssignmentOp(node.targets[0].id, AssignmentOp.none, [var], node)
            self.assignments.append(op)
            self.generic_visit(node)
            return
        # Handle cases where the rhs is a binary op
        if isinstance(node.value, ast.BinOp):
            assert isinstance(node.targets[0], ast.Name)
            assert isinstance(node.value.right, ast.Name)
            assert isinstance(node.value.left, ast.Name)
            if node.value.left.id not in self.name_to_var:
                left_var = Variable(node.value.left.id)
                self.name_to_var[node.value.left.id] = left_var
            else:
                left_var = self.name_to_var[node.value.left.id]
            if node.value.right.id not in self.name_to_var:
                right_var = Variable(node.value.right.id)
                self.name_to_var[node.value.right.id] = right_var
            else:
                right_var = self.name_to_var[node.value.right.id]
            assign_op = None
            if node.value.op == ast.Add():
                assign_op = AssignmentOp.add
            elif node.value.op == ast.Sub():
                assign_op = AssignmentOp.sub
            elif node.value.op == ast.Mult():
                assign_op = AssignmentOp.mul
            elif node.value.op == ast.Div():
                assign_op = AssignmentOp.div
            op = AssignmentOp(node.targets[0].id, assign_op, [left_var, right_var], node)
            self.assignments.append(op)
            self.generic_visit(node)
            return
        # Handle cases where shape is extracted from a tensor and assigned to variables
        if isinstance(node.targets[0], ast.Tuple):
            if isinstance(node.value, ast.Tuple):
                for lhs, rhs in zip(node.targets[0].elts, node.value.elts):
                    assert isinstance(lhs, ast.Name)
                    if isinstance(rhs, ast.Name):
                        if rhs.id not in self.name_to_var:
                            var = Variable(rhs.id)
                            self.name_to_var[rhs.id] = var
                        else:
                            var = self.name_to_var[rhs.id]
                        op = AssignmentOp(lhs.id, AssignmentOp.none, [var], node)
                        self.assignments.append(op)
                    elif isinstance(rhs, ast.Constant):
                        if type(rhs.value) == float:
                            self.func_args.append(self.val_defs[lhs.id])
                        self.val_defs[lhs.id] = Variable(lhs.id, rhs.value)
                        op = AssignmentOp(lhs.id, AssignmentOp.none, [Variable("", rhs.value)], node)
                        self.assignments.append(op)
                    elif isinstance(rhs, ast.BinOp):
                        assert isinstance(rhs.left, ast.Name)
                        assert isinstance(rhs.right, ast.Name)
                        if rhs.left.id not in self.name_to_var:
                            left_var = Variable(rhs.left.id)
                            self.name_to_var[rhs.left.id] = left_var
                        else:
                            left_var = self.name_to_var[rhs.left.id]
                        if rhs.right.id not in self.name_to_var:
                            right_var = Variable(rhs.right.id)
                            self.name_to_var[rhs.right.id] = right_var
                        else:
                            right_var = self.name_to_var[rhs.right.id]
                        assign_op = None
                        if rhs.op == ast.Add():
                            assign_op = AssignmentOp.add
                        elif rhs.op == ast.Sub():
                            assign_op = AssignmentOp.sub
                        elif rhs.op == ast.Mult():
                            assign_op = AssignmentOp.mul
                        elif rhs.op == ast.Div():
                            assign_op = AssignmentOp.div
                        op = AssignmentOp(lhs.id, assign_op, [left_var, right_var], node)
                        self.assignments.append(op)
                    else:
                        # Ignore the case where div_ceiling is called
                        if isinstance(rhs, ast.Call):
                            assert isinstance(rhs.func, ast.Name)
                            if rhs.func.id == "div_ceil":
                                self.skip_calls.add(rhs)
                            elif rhs.func.id == "min" or rhs.func.id == "max":
                                for arg in rhs.args:
                                    if isinstance(arg, ast.Constant):
                                        self.val_defs[lhs.id] = Variable(lhs.id, arg.value)
                                        #self.assignments[lhs.id] = arg.value
                                self.skip_calls.add(rhs)
                        else:
                            assert False, "unhandled assignment"
            elif isinstance(node.value, ast.Attribute):
                assert isinstance(node.value.value, ast.Name)
                name = node.value.value.id
                if node.value.attr == "shape":
                    if name in self.func_args:
                        shape_dims = list()
                        for elt in node.targets[0].elts:
                            assert isinstance(elt, ast.Name)
                            self.val_defs[elt.id] = Variable(elt.id)
                            shape_dims.append(self.val_defs[elt.id])
                        # Shape dimensions are represented as
                        # arguments to the function represnting semantics.
                        tensor = Tensor(name, shape_dims)
                        self.val_defs[name] = tensor
                        for idx, arg in enumerate(self.func_args):
                            if arg == tensor.name:
                                self.func_args[idx] = tensor
                                break
                    else:
                        if name in self.val_defs:
                            tensor = self.val_defs[name]
                            shape_dims = list()
                            for elt in node.targets[0].elts:
                                assert isinstance(elt, ast.Name)
                                self.val_defs[elt.id] = Variable(elt.id)
                                shape_dims.append(self.val_defs[elt.id])
                            tensor.shape = shape_dims
                            for idx, arg in enumerate(self.func_args):
                                if arg == tensor.name:
                                    self.func_args[idx] = tensor
                                    break
            self.generic_visit(node)
            return
        # Handle cases where tiles/tensors are defined as ndarrays
        if isinstance(node.value, ast.Call):           
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "ndarray":
                    assert isinstance(node.targets[0], ast.Name)
                    assert isinstance(node.value.args[0], ast.Tuple)
                    shape_dims = list()
                    for elt in node.value.args[0].elts:
                        if isinstance(elt, ast.Name):
                            self.val_defs[elt.id] = Variable(elt.id)
                            shape_dims.append(self.val_defs[elt.id])
                        elif isinstance(elt, ast.Constant):
                            shape_dims.append(elt.value)
                        else:
                            assert isinstance(elt, ast.Call)
                            assert isinstance(elt.func, ast.Name)
                            assert elt.func.id == "par_dim"
                            assert len(elt.args) == 1
                            if isinstance(elt.args[0], ast.Name):
                                self.val_defs[elt.args[0].id] = Variable(elt.args[0].id)
                                shape_dims.append(self.val_defs[elt.args[0].id])
                            else:
                                assert isinstance(elt.args[0], ast.Constant)
                                shape_dims.append(elt.args[0].id)
                            self.skip_calls.add(elt)
                    self.val_defs[node.targets[0]] = Tensor(node.targets[0].id, shape_dims)
                    self.skip_calls.add(node.value)
                    self.generic_visit(node)
                    return
                if node.value.func.attr == "ceil":
                    assert(node.value.args[0], ast.BinOp)
                    assert(node.targets[0], ast.Name)
                    assert(isinstance(node.value.args[0].left, ast.Name))
                    assert(isinstance(node.value.args[0].right, ast.Name))
                    if node.value.args[0].left.id not in self.name_to_var:
                        left_var = Variable(node.value.args[0].left.id)
                        self.name_to_var[node.value.args[0].left.id] = left_var
                    else:
                        left_var = self.name_to_var[node.value.args[0].left.id]
                    if node.value.args[0].right.id not in self.name_to_var:
                        right_var = Variable(node.value.args[0].right.id)
                        self.name_to_var[node.value.args[0].right.id] = right_var
                    else:
                        right_var = self.name_to_var[node.value.args[0].right.id]
                    #assert node.value.args[0].op == ast.Div()
                    assign_op = AssignmentOp.div
                    op = AssignmentOp(node.targets[0].id, assign_op, [left_var, right_var], node)
                    self.assignments.append(op)      
                self.generic_visit(node)
                return
            if node.value.attr == "dtypes":
                assert isinstance(node.value.value, ast.Name)
                name = node.value.value.id
                if name in self.func_args:
                    assert isinstance(node.targets[0], ast.Name)
                    # Data type is represnted as argument to the function 
                    # represnting semantics.
                    tensor = Tensor(name, dtype=Variable(node.targets[0].id))
                    self.val_defs[name] = tensor
                    for idx, arg in enumerate(self.func_args):
                        if arg == tensor.name:
                            self.func_args[idx] = tensor
                            break
                else:
                    if name in self.val_defs:
                        tensor = self.val_defs[name]
                        tensor.dtype = Variable(node.targets[0].id)
                        for idx, arg in enumerate(self.func_args):
                            if arg == tensor.name:
                                self.func_args[idx] = tensor
                                break
                self.generic_visit(node)
                return
        #assert False, "unhandled assignment"
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)

    def visit_Assert(self, node):
        # Extract relationships between variables in asserts and record the ones that we can.
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Call):
                lhs = node.test.left
                if isinstance(lhs.func, ast.Name):
                    if lhs.func.id == "tuple":
                        if isinstance(lhs.args[0], ast.Attribute):
                            assert isinstance(lhs.args[0].value, ast.Name)
                            name = lhs.args[0].value.id
                            if lhs.args[0].attr == "shape":
                                if isinstance(node.test.ops[0], ast.Eq):
                                    if isinstance(node.test.comparators[0], ast.Tuple):
                                        shape_dims = list()
                                        for dim in node.test.comparators[0].elts:
                                            assert isinstance(dim, ast.Name)
                                            if dim.id not in self.val_defs:
                                                self.val_defs[dim.id] = Variable(dim.id)
                                                self.name_to_var[dim.id] = self.val_defs[dim.id]
                                            shape_dims.append(self.val_defs[dim.id])
                                        if name not in self.val_defs:
                                            tensor = Tensor(name, shape_dims)
                                        else:
                                            tensor = self.val_defs[name]
                                            tensor.shape = shape_dims
                                        self.val_defs[name] = tensor
                                        for idx, arg in enumerate(self.func_args):
                                            if isinstance(arg, Tensor):
                                                if arg.name == tensor.name:
                                                    self.func_args[idx] = tensor
                                                    break
                                            else:
                                                if arg == tensor.name:
                                                    self.func_args[idx] = tensor
                                                    break
            else:
                assert isinstance(node.test.left, ast.Name)
                varname = node.test.left.id
                if varname not in self.val_defs:
                    self.val_defs[varname] = Variable(varname)
                if isinstance(node.test.comparators[0], ast.Constant):
                    constant_val = node.test.comparators[0].value
                    if isinstance(node.test.ops[0], ast.Eq):
                        self.val_defs[varname].real_val = constant_val
                        var = Variable("", constant_val)
                        self.assignments.append(AssignmentOp(varname, AssignmentOp.none, [var], node))
                    else:
                        self.val_defs[varname].constraints.append((node.test.ops[0], constant_val))
                else:
                    assert(isinstance(node.test.comparators[0], ast.Name))
                    if node.test.comparators[0].id not in self.val_defs:
                        self.val_defs[node.test.comparators[0].id] = Variable(node.test.comparators[0].id)
                    if isinstance(node.test.ops[0], ast.Eq):
                        self.val_defs[varname].real_val = self.val_defs[node.test.comparators[0].id]
                        rhs_name = node.test.comparators[0].id
                        if rhs_name not in self.name_to_var:
                            var = Variable(rhs_name)
                            self.name_to_var[rhs_name] = var
                        else:
                            var = self.name_to_var[rhs_name]
                        self.assignments.append(AssignmentOp(varname, AssignmentOp.none, [var], node))
                    else:
                        self.val_defs[varname].constraints.append((node.test.ops[0], self.val_defs[node.test.comparators[0].id]))
                        self.val_defs[node.test.comparators[0].id].constraints.append((node.test.ops[0], self.val_defs[varname]))
                self.name_to_var[varname] = self.val_defs[varname]
        elif isinstance(node.test, ast.BoolOp):
            if isinstance(node.test.op, (ast.Or, ast.And)):
                for value in node.test.values:
                    if isinstance(value, ast.Compare):
                        left = value.left
                        ops = value.ops
                        comparators = value.comparators
                        if isinstance(left, ast.Name):
                            if isinstance(comparators[0], ast.Constant):
                                constant_val = comparators[0].value
                                varname = left.id
                                if isinstance(ops[0], ast.Eq):
                                    if varname not in self.val_defs:
                                        self.val_defs[varname] = Variable(varname, constant_val)
                                    else:
                                        self.val_defs[varname].real_val = constant_val
                                    self.name_to_var[varname] = self.val_defs[varname]
                                else:
                                    if varname not in self.val_defs:
                                        self.val_defs[varname] = Variable(varname)
                                    self.val_defs[varname].constraints.append((ops[0], constant_val))
                                    self.name_to_var[varname] = self.val_defs[varname]
                        elif isinstance(left, ast.BinOp) and isinstance(left.left, ast.Name):
                            if isinstance(left.right, ast.Constant) and isinstance(comparators[0], ast.Constant):
                                if varname not in self.val_defs:
                                    self.val_defs[varname] = Variable(varname)
                                self.val_defs[varname].constraints.append((left.op, left.right.value, ops[0], comparators[0].value))
                                self.name_to_var[varname] = self.val_defs[varname]
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node not in self.skip_calls:
                if node.func.id in BaseSemaGenerator.calls_to_ignore:
                    self.generic_visit(node)
                    return
        if isinstance(node.func, ast.Attribute):
            if node not in self.skip_calls:
                if node.func.attr in BaseSemaGenerator.calls_to_ignore:
                    self.generic_visit(node)
                    return
        self.generic_visit(node)

    def visit_For(self, node):
        self.generic_visit(node)

    def visit_Expr(self, node):
        self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)

    def visit_Global(self, node):
        assert False, "Global statements are not yet supported in NKS"

    def visit_Nonlocal(self, node):
        assert False, "Nonlocal statements are not yet supported in NKS"

    def visit_AsyncFor(self, node):
        assert False, "Async for loops are not supported in NKS"

    def visit_Delete(self, node):
        assert False, "Delete statements are not supported in NKS"

    def visit_AsyncFunctionDef(self, node):
        assert False, "Async funtion def are not supported in NKS"

    def visit_ClassDef(self, node):
        assert False, "Classes are not supported in NKS"

    def visit_While(self, node):
        assert False, "While loops are not yet supported in NKS"

    def visit_With(self, node):
        assert False, "With statements are not yet supported in NKS"

    def visit_AsyncWith(self, node):
        assert False, "Async With statements are not supported in NKS"

    def visit_Raise(self, node):
        assert False, "Raise statements are not supported in NKS"

    def visit_Try(self, node):
        assert False, "Try statements are not supported in NKS"

    def visit_Import(self, node):
        assert False, "Import statements are not supported in NKS"

    def visit_ImportFrom(self, node):
        assert False, "Import from statements are not supported in NKS"

    def visit_Pass(self, node):
        assert False, "Pass statements are not yet supported in NKS"

    def visit_Break(self, node):
        assert False, "Break statements are not yet supported in NKS"

    def visit_Continue(self, node):
        assert False, "Continue statements are not yet supported in NKS"

    def generate_nki_op(self, prefix, name, node, opkind, args, keywords=dict(), elemwise_ops=list()):
        if opkind == "nc_matmul":
            stationary = keywords.get("stationary")
            if stationary == None:
                stationary = args[0]
                moving = keywords.get("moving")
                if moving == None:
                    moving = args[1]
            else:
                moving = keywords.get("moving")
                if moving == None:
                    moving = args[0]
            return NKI_nc_matmul(prefix, name, node, stationary, moving)
        if opkind == "nc_transpose":
            data = keywords["data"]
            if data == None:
                data = args[0]
                return NKI_nc_transpose(prefix, name, data, node)
        if opkind == "activation":
            data = keywords.get("data")
            if data == None:
                data = args[0]
                bias = keywords.get("bias")
                if bias == None:
                    if len(args) < 2:
                      bias = None
                    else:
                      bias = args[1]
                    scale = keywords.get("scale")
                    if scale == None:
                      if len(args) < 3:
                        scale = 1
                      else:
                        scale = args[2]
                else:
                  scale = keywords.get("scale")
                  if scale == None:
                    if len(args) < 2:
                      scale = 1
                    else:
                      scale = args[1]
            else:
                bias = keywords.get("bias")
                if bias == None:
                    if len(args) == 0:
                      bias = None
                    else:
                      bias = args[0]
                    scale = keywords.get("scale")
                    if scale == None:
                      if len(args) < 2:
                        scale = 1
                      else:
                        scale = args[1]
                else:
                  scale = keywords.get("scale")
                  if scale == None:
                    if len(args) == 0:
                      scale = 1
                    else:
                      scale = args[0]
                return NKI_activation(prefix, name, node, elemwise_ops, data, bias, scale)
        if opkind == "add":
            return NKI_add(prefix, name, node, args[0], args[1])
        if opkind == "subtract":
            return NKI_subtract(prefix, name, node, args[0], args[1]) 
        if opkind == "multiply":
            return NKI_multiply(prefix, name, node, args[0], args[1]) 
        if opkind == "divide":
            return NKI_divide(prefix, name, node, args[0], args[1])
        if opkind == "maximum":
            return NKI_maximum(prefix, name, node, args[0], args[1]) 
        if opkind == "minimum":
            return NKI_minimum(prefix, name, node, args[0], args[1]) 
        if opkind == "max":
            axis = keywords.get("axis")
            if axis == None:
                axis = args[1]
            return NKI_max(prefix, name, node, args[0], axis)
        if opkind == "min":
            axis = keywords.get("axis")
            if axis == None:
                axis = args[1]
            return NKI_min(prefix, name, node, args[0], axis)
        if opkind == "sum":
            axis = keywords.get("axis")
            if axis == None:
                axis = args[1]
            return NKI_sum(prefix, name, node, args[0], axis)
        if opkind == "prod":
            axis = keywords.get("axis")
            if axis == None:
                axis = args[1]
            return NKI_prod(prefix, name, node, args[0], axis)
        if opkind == "negative":
            return NKI_negative(prefix, name, node, args[0])
        if opkind == "exp":
            return NKI_exp(prefix, name, node, args[0])
        if opkind == "log":
            return NKI_log(prefix, name, node, args[0])
        if opkind == "sin":
            return NKI_sin(prefix, name, node, args[0])
        if opkind == "cos":
            return NKI_cos(prefix, name, node, args[0])
        if opkind == "tan":
            return NKI_tan(prefix, name, node, args[0]) 
        if opkind == "tanh":
            return NKI_tanh(prefix, name, node, args[0])
        if opkind == "arctan":
            return NKI_arctan(prefix, name, node, args[0]) 
        if opkind == "sqrt":
            return NKI_sqrt(prefix, name, node, args[0])
        if opkind == "rsqrt":
            return NKI_rsqrt(prefix, name, node, args[0])
        if opkind == "relu":
            return NKI_relu(prefix, name, node, args[0]) 
        if opkind == "softplus":
            return NKI_softplus(prefix, name, node, args[0])
        if opkind == "mish":
            return NKI_mish(prefix, name, node, args[0]) 
        if opkind == "square":
            return NKI_square(prefix, name, node, args[0]) 
        if opkind == "reciprocal":
            return NKI_reciprocal(prefix, name, node, args[0])
        if opkind == "softmax":
            bias = keywords.get("bias")
            if bias == None:
                bias = args[1]
            return NKI_softmax(prefix, name, node, args[0], bias)
        if opkind == "matmul":
            stationary = keywords.get("stationary")
            if stationary == None:
                stationary = args[0]
                moving = keywords.get("moving")
                if moving == None:
                    moving = args[1]
                    transpose_x = keywords.get("transpose_x")
                    if transpose_x == None:
                        if len(args) < 3:
                          transpose_x = False
                        else:
                          transpose_x = args[2]
                else:
                    transpose_x = keywords.get("transpose_x")
                    if transpose_x == None:
                      if len(args) < 2:
                        transpose_x = False
                      else:
                        transpose_x = args[1]
            else:
                moving = keywords.get("moving")
                if moving == None:
                    moving = args[0]
                    transpose_x = keywords.get("transpose_x")
                    if transpose_x == None:
                        if len(args) < 2:
                          transpose_x = False
                        else:
                          transpose_x = args[1]
                else:
                    transpose_x = keywords.get("transpose_x")
                    if transpose_x == None:
                        if len(args) == 0:
                          transpose_x = False
                        else:
                          transpose_x = args[0]
            return NKI_matmul(prefix, name, node, stationary, moving, transpose_x)
        if opkind == "transpose":
            data = keywords.get("data")
            if data == None:
                data = args[0]
                return NKI_transpose(prefix, name, data, node)

    def generate_torch_op(self, prefix, name, node, opkind, args, keywords=dict()):
        if opkind == "add":
            return Torch_add(prefix, name, node, args[0], args[1])
        if opkind == "subtract":
            return Torch_subtract(prefix, name, node, args[0], args[1])
        if opkind == "multiply":
            return Torch_multiply(prefix, name, node, args[0], args[1])
        if opkind == "divide":
            return Torch_divide(prefix, name, node, args[0], args[1])
        if opkind == "maximum":
            return Torch_maximum(prefix, name, node, args[0], args[1])
        if opkind == "minimum":
            return Torch_minimum(prefix, name, node, args[0], args[1])
        if opkind == "max":
            axis = keywords.get("dim")
            if axis is None:
                axis = args[1]
            return Torch_max(prefix, name, node, args[0], axis)
        if opkind == "min":
            axis = keywords.get("dim")
            if axis is None:
                axis = args[1]
            return Torch_min(prefix, name, node, args[0], axis)
        if opkind == "sum":
            axis = keywords.get("dim")
            if axis is None:
                axis = args[1]
            return Torch_sum(prefix, name, node, args[0], axis)
        if opkind == "prod":
            axis = keywords.get("dim")
            if axis is None:
                axis = args[1]
            return Torch_prod(prefix, name, node, args[0], axis)
        if opkind == "negative":
            return Torch_negative(prefix, name, node, args[0])
        if opkind == "exp":
            return Torch_exp(prefix, name, node, args[0])
        if opkind == "log":
            return Torch_log(prefix, name, node, args[0])
        if opkind == "sin":
            return Torch_sin(prefix, name, node, args[0])
        if opkind == "cos":
            return Torch_cos(prefix, name, node, args[0])
        if opkind == "tan":
            return Torch_tan(prefix, name, node, args[0])
        if opkind == "tanh":
            return Torch_tanh(prefix, name, node, args[0])
        if opkind == "arctan":
            return Torch_arctan(prefix, name, node, args[0])
        if opkind == "sqrt":
            return Torch_sqrt(prefix, name, node, args[0])
        if opkind == "rsqrt":
            return Torch_rsqrt(prefix, name, node, args[0])
        if opkind == "relu":
            return Torch_relu(prefix, name, node, args[0])
        if opkind == "softplus":
            return Torch_softplus(prefix, name, node, args[0])
        if opkind == "mish":
            return Torch_mish(prefix, name, node, args[0])
        if opkind == "square":
            return Torch_square(prefix, name, node, args[0])
        if opkind == "softmax":
            axis = keywords.get("dim")
            if axis is None:
                axis = args[1]
            return Torch_softmax(prefix, name, node, args[0], axis)
        if opkind == "matmul":
            return Torch_matmul(prefix, name, node, args[0], args[1])
        if opkind == "transpose":
            return Torch_transpose(prefix, name, node, args[0])


class SpecSemaGenerator(BaseSemaGenerator):
    def __init__(self, ast_tree):
        super().__init__(ast_tree)

    def visit_Assign(self, node):
        # Handle cases where hole operations and NKI operations are used
        if isinstance(node.value, ast.Call):
            if node.value.func.attr not in BaseSemaGenerator.calls_to_ignore:
                operands = list()
                for arg in node.value.args:
                  operands.append(self._parse_arg(arg))
                keywords_mapping = dict()
                for keyword in node.value.keywords:
                    assert isinstance(keyword, ast.keyword)
                    assert isinstance(keyword.value, (ast.Name, ast.Constant))
                    keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
                assert isinstance(node.value.func.value, ast.Name)
                lhs = self._get_lhs(node.targets[0])
                # Extract the op prefix and the kind of operation
                prefix = node.value.func.value.id
                opkind = node.value.func.attr
                op = self.generate_torch_op(prefix, lhs, node, \
                                            opkind, operands, keywords_mapping)
                self.statements.append(op)
                self.val_defs[lhs] = op
            self.generic_visit(node)
            return
        #super().visit_Assign(node)
        self.generic_visit(node)


class LoopLessSemaGenerator(BaseSemaGenerator):
    def __init__(self, ast_tree):
        super().__init__(ast_tree)

    def visit_Assign(self, node):
        # Handle cases where hole operations and NKI operations are used
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "hole_op":
                    operands = list()
                    for arg in node.value.args:
                        operands.append(self._parse_arg(arg, keywords=node.value.keywords))
                    lhs = self._get_lhs(node.targets[0])
                    op = Hole_op(lhs, operands, node)
                    self.statements.append(op)
                    self.val_defs[lhs] = op
            elif node.value.func.attr not in BaseSemaGenerator.calls_to_ignore:
                elemwise_ops = list()
                operands = list()
                for arg in node.value.args:
                  # One of the arguments could be an op
                  if isinstance(arg, ast.Attribute):
                    elemwise_ops.append(arg.attr)
                    continue
                  operands.append(self._parse_arg(arg))
                keywords_mapping = dict()
                for keyword in node.value.keywords:
                    assert isinstance(keyword, ast.keyword)
                    if isinstance(keyword.value, (ast.Name, ast.Subscript)):
                        keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
                    else:
                        assert isinstance(keyword.value, ast.Constant)
                        keywords_mapping[keyword.arg] = keyword.value.value
                assert isinstance(node.value.func.value, ast.Name)
                lhs = self._get_lhs(node.targets[0])
                # Extract the op prefix and the kind of operation
                prefix = node.value.func.value.id
                opkind = node.value.func.attr
                op = self.generate_nki_op(prefix, lhs, node, opkind, \
                                    operands, keywords_mapping, elemwise_ops=elemwise_ops)
                self.statements.append(op)
                self.val_defs[lhs] = op
            self.generic_visit(node)
            return
        if isinstance(node.value, ast.BinOp):
            # Check if this is actually a tensor operation
            if isinstance(node.targets[0], ast.Subscript) \
            or isinstance(node.value.left, ast.Subscript) \
            or isinstance(node.value.right, ast.Subscript):
                # This is a tensor operation
                prefix = "nki.language."
                if isinstance(node.value.op, ast.Mult):
                    opkind = "multiply"
                elif isinstance(node.value.op, ast.Add):
                    opkind = "add"
                # Account for the left operand
                operands = list()
                operands.append(self._parse_arg(node.value.left))
                operands.append(self._parse_arg(node.value.right))
                lhs = self._get_lhs(node.targets[0])
                op = self.generate_nki_op(prefix, lhs, node, opkind, operands)
                self.statements.append(op)
                self.val_defs[lhs] = op
                self.generic_visit(node)
                return
        super().visit_Assign(node)
        
    def visit_AugAssign(self, node):
        if isinstance(node.value, ast.Call):
            operands = list()
            for arg in node.value.args:
                operands.append(self._parse_arg(arg))
            keywords_mapping = dict()
            for keyword in node.value.keywords:
                assert isinstance(keyword, ast.keyword)
                if isinstance(keyword.value, ast.Subscript):
                    keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
                else:
                    assert isinstance(keyword.value, ast.Name)
                    keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
            assert isinstance(node.value.func.value, ast.Name)
            lhs = self._get_lhs(node.target)
            # Extract the op prefix and the kind of operation
            prefix = node.value.func.value.id
            opkind = node.value.func.attr
            op = self.generate_nki_op(prefix, lhs, node, \
                                        opkind, operands, keywords_mapping)
            self.statements.append(op)
            self.val_defs[lhs] = op
        self.generic_visit(node)


class OrgSemaGenerator(BaseSemaGenerator):
    def __init__(self, ast_tree, hole_defs_map, replacement_vals):
        self.loop_stack = list()
        self.nested_loops = list()
        self.node_to_loop = dict()
        self.hole_defs_map = hole_defs_map
        super().__init__(ast_tree)
        self.replacement_vals = replacement_vals

    def _parse_subcript_arg(self, arg, node):
        if isinstance(arg, ast.Subscript):
            assert isinstance(arg.value, ast.Name)
            # This expression indexes into a tensor, so we need to
            # generate a hole operation that is extracts a slice of the tensor.
            operand = arg.value.id
            if operand in self.replacement_vals:
                assert self.replacement_vals[operand] in self.val_defs
                operand = self.val_defs[self.replacement_vals[operand]]
            else:
                operand = self.val_defs[arg.value.id]
            hole_op = Hole_op("ext_" + operand.name, [operand], None)
            return hole_op
        return None

    def visit_For(self, node):
        assert(isinstance(node.target, ast.Name))
        assert isinstance(node.iter, ast.Call)
        if any(isinstance(parent, ast.For) for parent in self.loop_stack):
            self.nested_loops.append(node)
        if isinstance(node.iter.args[0], ast.Name):
            nks_loop = NKS_for_loop(node, node.target.id, node.iter.args[0].id)
        else:
            assert(isinstance(node.iter.args[0], ast.Constant))
            nks_loop = NKS_for_loop(node, node.target.id, node.iter.args[0].value)
        self.loop_stack.append(nks_loop)
        self.node_to_loop[node] = nks_loop
        self.generic_visit(node)
        loop = self.loop_stack.pop()
        loop.ret_val = loop.contents[-1]
        if isinstance(loop.ret_val, NKS_for_loop):
            loop.ret_val = loop.ret_val.ret_val
        if len(self.loop_stack) == 1:
            loop.parent = self.loop_stack[0]
        elif len(self.loop_stack) > 1:
            loop.parent = self.loop_stack[-1]
        else:
            loop.parent = None
        #loop.parent = loop_stack[-1] if len(self.loop_stack) > 0 else None
        if len(self.loop_stack) == 0:
            self.statements.append(loop)
        else:
            self.loop_stack[-1].contents.append(loop)

    def visit_Assign(self, node):
        # Handle cases where hole operations and NKI operations are used
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "hole_op":
                    lhs = self._get_lhs(node.targets[0])
                    operands = list()
                    for arg in node.value.args:
                        hole_op = self._parse_subcript_arg(arg, node)
                        if hole_op != None:
                            operands.append(hole_op)
                            if len(self.loop_stack) == 0:
                                self.statements.append(hole_op)
                            else:
                                self.loop_stack[-1].contents.append(hole_op)
                        else:
                            operands.append(self._parse_arg(arg, keywords=node.value.keywords))
                    for hole_op, hole_def in self.hole_defs_map.items():
                        if node != hole_op.ast_expr:
                            continue
                        assert isinstance(hole_def, TensorOp)
                        op = hole_def
                        # Parse the arguments of this operation to see if
                        # we need to replace it with a hole operation
                        if isinstance(hole_def, (NKS_assign, NKS_transpose)):
                            op.operands = operands
                        elif isinstance(hole_def, (NKS_reduce, NKS_broadcast)):
                            op.operands[0] = operands[0]                            
                        if len(self.loop_stack) == 0:
                            self.statements.append(op)
                        else:
                            self.loop_stack[-1].contents.append(op)
                        self.val_defs[lhs] = op
                self.generic_visit(node)
                return
            elif node.value.func.attr not in BaseSemaGenerator.calls_to_ignore:
                elemwise_ops = list()
                operands = list()
                for arg in node.value.args:
                    # One of the arguments could be an op
                    if isinstance(arg, ast.Attribute):
                        elemwise_ops.append(arg.attr)
                        continue
                    hole_op = self._parse_subcript_arg(arg, node)
                    if hole_op != None:
                        operands.append(hole_op)
                        if len(self.loop_stack) == 0:
                            self.statements.append(hole_op)
                        else:
                            self.loop_stack[-1].contents.append(hole_op)
                    else:
                        operands.append(self._parse_arg(arg))
                keywords_mapping = dict()
                for keyword in node.value.keywords:
                    assert isinstance(keyword, ast.keyword)
                    if isinstance(keyword.value, (ast.Name, ast.Subscript)):
                        hole_op = self._parse_subcript_arg(keyword.value, node)
                        if hole_op != None:
                            #operands.append(hole_op)
                            keywords_mapping[keyword.arg] =  hole_op
                            if len(self.loop_stack) == 0:
                                self.statements.append(hole_op)
                            else:
                                self.loop_stack[-1].contents.append(hole_op)
                        else:
                            keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
                    else:
                        assert isinstance(keyword.value, ast.Constant)
                        keywords_mapping[keyword.arg] = keyword.value.value
                assert isinstance(node.value.func.value, ast.Name)
                # Extract the op prefix and the kind of operation
                prefix = node.value.func.value.id
                opkind = node.value.func.attr
                lhs = self._get_lhs(node.targets[0])
                op = self.generate_nki_op(prefix, lhs, node, opkind, operands, \
                                    keywords_mapping, elemwise_ops=elemwise_ops)
                if len(self.loop_stack) == 0:
                    self.statements.append(op)
                else:
                    self.loop_stack[-1].contents.append(op)
                self.val_defs[lhs] = op
                self.generic_visit(node)
                return
            super().visit_Assign(node)
            return
        if isinstance(node.value, ast.BinOp):
            # Check if this is actually a tensor operation
            if isinstance(node.targets[0], ast.Subscript) \
            or isinstance(node.value.left, ast.Subscript) \
            or isinstance(node.value.right, ast.Subscript):
                # This is a tensor operation
                prefix = "nki.language."
                if isinstance(node.value.op, ast.Mult):
                    opkind = "multiply"
                elif isinstance(node.value.op, ast.Add):
                    opkind = "add"
                # Account for the left operand
                lhs = self._get_lhs(node.targets[0])
                operands = list()
                hole_op = self._parse_subcript_arg(node.value.left, node)
                if hole_op != None:
                    operands.append(hole_op)
                    if len(self.loop_stack) == 0:
                        self.statements.append(hole_op)
                    else:
                        self.loop_stack[-1].contents.append(hole_op)
                else:
                    operands.append(self._parse_arg(node.value.left))
                hole_op = self._parse_subcript_arg(node.value.right, node)
                if hole_op != None:
                    operands.append(hole_op)
                    if len(self.loop_stack) == 0:
                        self.statements.append(hole_op)
                    else:
                        self.loop_stack[-1].contents.append(hole_op)
                else:
                    operands.append(self._parse_arg(node.value.right))
                op = self.generate_nki_op(prefix, lhs, node, opkind, operands)
                if len(self.loop_stack) == 0:
                    self.statements.append(op)
                else:
                    self.loop_stack[-1].contents.append(op)
                self.val_defs[lhs] = op
                self.generic_visit(node)
                return
        super().visit_Assign(node)

    def visit_AugAssign(self, node):
        if isinstance(node.value, ast.Call):
            lhs = self._get_lhs(node.target)
            operands = list()
            for arg in node.value.args:
                hole_op = self._parse_subcript_arg(arg, node)
                if hole_op != None:
                    operands.append(hole_op)
                    if len(self.loop_stack) == 0:
                        self.statements.append(hole_op)
                    else:
                        self.loop_stack[-1].contents.append(hole_op)
                else:
                    operands.append(self._parse_arg(arg))
            keywords_mapping = dict()
            for keyword in node.value.keywords:
                assert isinstance(keyword, ast.keyword)
                if isinstance(keyword.value, ast.Subscript):
                    hole_op = self._parse_subcript_arg(keyword.value, node)
                    if hole_op != None:
                        #operands.append(hole_op)
                        keywords_mapping[keyword.arg] =  hole_op
                        if len(self.loop_stack) == 0:
                            self.statements.append(hole_op)
                        else:
                            self.loop_stack[-1].contents.append(hole_op)
                    else:
                        keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
                else:
                    assert isinstance(keyword.value, ast.Name)
                    keywords_mapping[keyword.arg] = self._parse_arg(keyword.value)
            assert isinstance(node.value.func.value, ast.Name)
            # Extract the op prefix and the kind of operation
            prefix = node.value.func.value.id
            opkind = node.value.func.attr
            op = self.generate_nki_op(prefix, lhs, node, \
                                        opkind, operands, keywords_mapping)
            if len(self.loop_stack) == 0:
                self.statements.append(op)
            else:
                self.loop_stack[-1].contents.append(op)
            self.val_defs[lhs] = op
        self.generic_visit(node)
