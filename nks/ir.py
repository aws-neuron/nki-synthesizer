"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Intermediate representation for NKS.

"""


from enum import auto


# Generate SSA value for NKS IR.
ssa_val = -1
def gen_name():
    global ssa_val
    ssa_val += 1
    return "%" + str(ssa_val)


# Some custom ops that we had to define because Rosette does not have
# built-in support for them.
def is_custom_op(op):
    custom_op = {"nks::var-exp", "nks::var-log", "nks::var-softplus", 
                "nks::var-mish", "nks::var-relu", "nks::var-sin", 
                "nks::var-cos", "nks::var-tan", "nks::var-tanh", 
                "nks::var-arctan", "nks::var-sqrt", "nks::var-rsqrt"}
    if op in custom_op:
        return True
    return False


class Variable:
    def __init__(self, name : str, real_val=None, constraints=list()):
        assert isinstance(name, str)
        self.name = name
        # We keep track of any real value that may be assigned to this variable
        self.real_val = real_val
        # We also keep track of any constraints that apply to this variable
        self.constraints = constraints
        
    def __eq__(self, value):
        if isinstance(value, Variable):
            return self.real_val == value.real_val and self.name == value.name
        return self.real_val == value
    
    def __ne__(self, value):
        return not self.__eq__(value)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        if self.real_val == None:
            return self.name
        return str(self.real_val)


class Tensor:
    def __init__(self, name : str, shape : list = None, dtype = None):
        assert isinstance(name, str)
        if isinstance(shape, list):
            for dim in shape:
                assert isinstance(dim, int) or isinstance(dim, Variable)
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.bitwidth = self._get_type_bitwidth()
        self.partial_stmts = list()
        self.partial_elems = list()

    def _get_type_bitwidth(self):
        if self.dtype == None:
            return 8 
        if "8" in self.dtype:
            return 8
        if "16" in self.dtype:
            return 16
        if "32" in self.dtype:
            return 32
        if "64" in self.dtype:
            return 64

    def __eq__(self, value):
        if isinstance(value, str):
            return self.name == value
        if not isinstance(value, Tensor):
            return False
        return self.name == value.name \
            and self.shape == value.shape \
            and self.dtype == value.dtype
            
    def __ne__(self, value):    
        return not self.__eq__(value)

    def __hash__(self):
        return hash(self.name)
    
    def row_major_access(self, num_rows, num_cols):
        # Upack some of the elements of the tensor
        length = self.bitwidth * num_rows * num_cols
        string = "(define-values ( "
        for _ in range(num_rows * num_cols):
            name = gen_name()
            string += name + " "
            self.partial_elems.append(name)
        string += ") (nks::var-unpack " + self.name + " #:prec " + str(self.bitwidth) \
                    + " #:len " + str(length) + "))"
        self.partial_stmts.append(string)


# Base op to represent a NKI language or an ISA op.
class TensorOp:
    def __init__(self, prefix : str, name : str, ast_expr, opkind : str, \
                        operands : list, elemwise_ops = list()):
        assert isinstance(prefix, str)
        assert isinstance(name, str)
        assert isinstance(operands, list)
        assert isinstance(opkind, str)
        self.name = name
        self.prefix = prefix
        self.opkind = opkind
        self.operands = operands
        self.ast_expr = ast_expr
        self.elemwise_ops = elemwise_ops
        self.bitwidth = None
        for operand in self.operands:
            if isinstance(operand, (Tensor, TensorOp)):
                self.bitwidth = operand.bitwidth
                break
        assert self.bitwidth != None
        self.partial_stmts = list()
        self.partial_elems = list()
    
    def row_major_access(self, x_name, rows, cols):
        # Upack some of the elements of the tensor
        length = self.bitwidth * rows * cols
        stmts = list()
        elems = list()
        string = "(define-values ( "
        for _ in range(rows * cols):
            name = gen_name()
            string += name + " "
            elems.append(name)
        string += ") (nks::var-unpack " + self.name + " #:prec " + str(self.bitwidth) \
                    + " #:len " + str(length) + "))"
        stmts.append(string)
        return elems, stmts

    def col_major_access(self, x_name, rows, cols):
        length = self.bitwidth * rows * cols
        stmts = list()
        elems = list()
        for j in range(cols):
            for i in range(rows):
                name = gen_name()
                low = (i * cols * self.bitwidth) + (j * self.bitwidth)
                high = low + self.bitwidth - 1
                stmts.append("(define " + name + " (nks::extract " + x_name + " " + str(low) \
                        + " #:prec " + str(self.bitwidth) + " #:len " + str(length) + "))")
                elems.append(name)
        return elems, stmts

    def get_elemwise_op(self, index : int):
        assert index < len(self.elemwise_ops)
        op = self.elemwise_ops[index]
        if "exp" in op:
            return "nks::var-exp"
        if "log" in op:
            return "nks::var-log"
        if "sin" in op:
            return "nks::var-sin"
        if "cos" in op:
            return "nks::var-cos"
        if "arctan" in op:
            return "nks::var-arctan"
        if "tanh" in op:
            return "nks::var-tanh"
        if "tan" in op:
            return "nks::var-tan"
        if "mish" in op:
            return "nks::var-mish"
        if "relu" in op:
            return "nks::var-relu"
        if "softplus" in op:
            return "nks::var-softplus"
        if "rsqrt" in op:
            return "nks::var-rsqrt"
        if "sqrt" in op:
            return "nks::var-sqrt"
        if "square" in op:
            return "nks::var-square"
        return None
    
    def __str__(self) -> str:
        string = self.name + " = " + self.opkind + "("
        for op in self.elemwise_ops:
            string += op + ", "
        for idx, operand in enumerate(self.operands):
            if isinstance(operand, (Tensor, Variable, TensorOp)):
                string += operand.name
            else:
                string += str(operand)
            if idx != len(self.operands) - 1:
                string += ", "
        string += ")"
        return string

    def __hash__(self):
        return hash(self.name)

    def constituent_ops(self):  
        return {}
    
    def reduction_ops(self):
        return {}
    
    def get_output_shape(self):
        NotImplemented

    # Subclassses must impleemnt this
    def to_rosette(self, space=0):
        NotImplemented


class AssignmentOp:
    add = auto()
    sub = auto()
    mul = auto()
    div = auto()
    mod = auto()
    none = auto()
    
    def __init__(self, name : str, op, operands : list, ast_expr):
        assert isinstance(name, str)
        assert isinstance(operands, list)
        assert len(operands) != 0
        assert op == AssignmentOp.add or \
            op == AssignmentOp.sub or \
            op == AssignmentOp.mul or \
            op == AssignmentOp.div or \
            op == AssignmentOp.mod or \
            op == AssignmentOp.none
        self.name = name
        self.operands = operands
        self.ast_expr = ast_expr
        self.op = op
        
    def __str__(self, space=0):
        indent = " " * space
        string = indent + self.name + " = "
        if self.op == AssignmentOp.add:
            string += "add "
        elif self.op == AssignmentOp.sub:
            string += "sub "
        elif self.op == AssignmentOp.mul:   
            string += "mul "
        elif self.op == AssignmentOp.div:
            string += "div "
        elif self.op == AssignmentOp.mod:
            string += "mod "
        elif self.op == AssignmentOp.none:
            string += ""
        else:
            assert False, "Invalid assignment op"
        for idx, operand in enumerate(self.operands):
            if operand.real_val == None:
                string += operand.name
            else:
                string += str(operand.real_val)
            if idx != len(self.operands) - 1:
                string += ", "
        return string
    
    def to_rosette(self, space=0):
        indent = " " * space
        string = indent + "(define " + self.name + " ("
        if self.op == AssignmentOp.add:
            string += "+ "
        elif self.op == AssignmentOp.sub:
            string += "- "
        elif self.op == AssignmentOp.mul:   
            string += "* "
        elif self.op == AssignmentOp.div:
            string += "/ "
        elif self.op == AssignmentOp.mod:
            string += "mod"
        elif self.op == AssignmentOp.none:
            string += ""
        else:
            assert False, "Invalid assignment op"
        for idx, operand in enumerate(self.operands):
            if operand.real_val == None:
                string += operand.name + " "
            else:
                string += str(operand.real_val) + " "
        string += ")"
        return string
        
    
class Hole_op(TensorOp):
    hole_number = 0

    def __init__(self, name : str, operands : list, ast_expr):
        assert len(operands) == 1
        opkind = "hole." + str(Hole_op.hole_number)
        Hole_op.hole_number += 1
        super().__init__("", name, ast_expr, opkind, operands)

    def to_rosette(self, space=0):
        string = " " * space + "(define " + self.name + " (" + self.opkind + " " + \
                 " ".join(operand.name if isinstance(operand, (Tensor, Variable)) \
                                    else str(operand) for operand in self.operands) + "))"
        return string


class NKS_transpose(TensorOp):
    def __init__(self, name, data):
        super().__init__("", name, None, "NKS_transpose", [data])

    def to_rosette(self, space=0):
        string = " " * space + "(define " + self.name + " (nks::mtx-transpose " + \
                 self.operands[0].name + " #:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class NKS_reduce(TensorOp):
    def __init__(self, name, op, data, axis):
            super().__init__("", name, None, "NKS_reduce", \
                              [data, axis], [op])

    def to_rosette(self, space=0):
        string = " " * space + "(define " + self.name + " (nks::mtx-reduce " + \
                 self.elemwise_ops[0] + " " + self.operands[0].name + " #:axis " + \
                 str(self.operands[1]) + " #:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class NKS_broadcast(TensorOp):
    def __init__(self, name, data, axis, num_reps):
        super().__init__("", name, None, "NKS_broadcast", [data, axis, num_reps])

    def to_rosette(self, space=0):
        string = " " * space + "(define " + self.name + " (nks::mtx-broadcast " + \
                 self.operands[0].name + " #:axis " + str(self.operands[1]) + " #:reps " + \
                 str(self.operands[2]) + "))"
        return string


class NKS_assign(TensorOp):
    def __init__(self, name, data):
        super().__init__("", name, None, "NKS_assign", [data])

    def to_rosette(self, space=0):
        string = " " * space + "(define " + self.name + " " + self.operands[0].name + ")"
        return string


class NKS_scan(TensorOp):
    def __init__(self, name, data0, data1, initial, reverse0, reverse1, op0, op1):
        super().__init__("", name, None, "NKS_scan", [data0, data1, initial, reverse0, reverse1], [op0, op1])

    def to_rosette(self, space=0):
        indent = " " * space
        string = f"{indent}(define {self.name} (nks::mtx-scan {self.elemwise_ops[0]} {self.elemwise_ops[1]} " \
                 f"{self.operands[0].name} {self.operands[1].name} {self.operands[2].name} " \
                 f"#:reverse0 {self.operands[3]} #:reverse1 {self.operands[4]} " \
                 f"#:prec {self.operands[0].bitwidth}))"
        return string


class NKS_extract_slice(TensorOp):
    def __init__(self, name, data, row_idx, col_idx, rows, cols):
        super().__init__("", name, None, "NKS_extract_slice", [data, row_idx, col_idx, rows, cols])

    def to_rosette(self, space=0):
        indent = " " * space
        string = f"{indent}(define {self.name} (nks::mtx-extract {self.operands[0].name} " \
                 f"#:row-idx {self.operands[1]} #:col-idx {self.operands[2]} " \
                 f"#:rows {self.operands[3]} #:cols {self.operands[4]}))"
        return string


class NKS_for_loop:
    colwise_concat = auto()
    rowwise_concat = auto()
    accumulate = auto()

    def __init__(self, ast_expr, iterator_name, end, ret_val=None, parent_loop=None):
        self.iterator = Variable(iterator_name)
        if isinstance(end, str):
            self.end = Variable(end)
        else:
            assert isinstance(end, int)
            self.end = end
        self.ret_val = ret_val
        self.contents = list()
        self.parent = parent_loop
        self.loop_kind = None
        self.is_tiled = False
        self.ast_expr = ast_expr
        
    @property
    def parents(self):
        if self.parent is None:
            return None
        parents = list()
        parent = self.parent
        while parent is not None:
            assert isinstance(parent, NKS_for_loop)
            parents.append(parent)
            parent = parent.parent
        # Innermost to outermost
        return parents

    def __eq__(self, other):
        if not isinstance(other, NKS_for_loop):
            return False
        return self.iterator == other.iterator and self.end == other.end \
                and self.contents == other.contents and self.parent == other.parent \
                and self.ret_val == other.ret_val and self.loop_kind == other.loop_kind

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)

    def __str__(self, num_space=0):
        string = ""
        spaces = " " * num_space
        loop_end = self.end.name if isinstance(self.end, Variable) else str(self.end)
        if self.loop_kind is not None:
            header = f"for_{self.loop_kind.name} ([{self.iterator.name} (range {loop_end})])"
        else:
            header = f"for ([{self.iterator.name} (range {loop_end})])"
        string += spaces + header + " {\n"
        for content in self.contents:
            if isinstance(content, NKS_for_loop):
                string += content.__str__(num_space + 2) + "\n"
            else:
                body_space = " " * (num_space + 2)
                string += body_space + str(content) + "\n"
        string += (spaces + "}")
        return string

    def to_rosette(self, space=0):
        indent = " " * space
        concat_kind = None
        assert self.loop_kind != None
        if self.loop_kind != NKS_for_loop.accumulate:
            if self.loop_kind == NKS_for_loop.colwise_concat:
                concat_kind = "nks::mtx-colwise-concat"
            elif self.loop_kind == NKS_for_loop.rowwise_concat:
                concat_kind = "nks::mtx-rowwise-concat"
            string = f"{indent}(define {self.retval.name} \n" \
                    + f"(apply {concat_kind} " \
                    + f"(for/list [{self.iterator.name} (nks::range {self.end})])\n"
        else:
            string = f"{indent}(define {self.retval.name}.tiles \n" \
                    + f"(for/list [{self.iterator.name} (nks::range {self.end})])\n"
        for content in self.contents:
            string += content.to_rosette(space + 2) + "\n"
        string += f"))){indent})"
        if self.loop_kind == NKS_for_loop.accumulate:
            string += f"{indent})(define {self.retval.name} \
                        (nks::mtx-accumulate {self.retval.name}.tiles))" 
        return string
