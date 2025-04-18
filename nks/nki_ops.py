"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Translation layer between Python to Rosette and vice versa.

"""


from ir import *


class NKI_nc_matmul(TensorOp):
    def __init__(self, prefix, name, ast_expr, stationary, moving):
        super().__init__(prefix, name, ast_expr, "NKI_nc_matmul", [stationary, moving])

    def generate_partial_ops(self, x_elems, y_elems, m, n, k):
        for i in range(m):
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} (nks::var-add{}))".format(
                    elem_name,
                    "".join(" (nks::var-mul {} {})".format(
                    x_elems[(o * m) + i], y_elems[(o * n) + j]) for o in range(k))
                )
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
    
    def constituent_ops(self):
        return {"nks::var-add", "nks::var-mul"}
    
    def reduction_ops(self):
        return {"nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.nc_matmul {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_nc_transpose(TensorOp):
    def __init__(self, prefix, name, ast_expr, data):
        super().__init__(prefix, name, ast_expr, "NKI_nc_transpose", [data])

    def generate_partial_ops(self, x_elems, m, n):
        for j in range(n):
            for i in range(m):
                self.partial_elems.append(x_elems[(i * n) + j])

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.nc_transpose {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_activation(TensorOp):
    def __init__(self, prefix, name, ast_expr, ops, data, bias=None, scale=1):
        if bias is None:
            super().__init__(prefix, name, ast_expr, "NKI_activation", [data, int(scale)], ops)
        else:
            super().__init__(prefix, name, ast_expr, "NKI_activation", [data, bias, int(scale)], ops)

    def generate_partial_ops(self, x_elems, m, n, bias_elems=None):
        if bias_elems is None:
            assert len(self.operands) == 2
            for i in range(m * n):
                subname = gen_name()
                if is_custom_op(self.get_elemwise_op(0)):
                    self.partial_stmts.append("(define {} ({} (nks::var-mul {} (bv {} {})) #:len {}))".format(
                                            subname, self.get_elemwise_op(0), x_elems[i], self.operands[1], \
                                            self.operands[0].bitwidth, self.operands[0].bitwidth))
                else:
                    self.partial_stmts.append("(define {} ({} (nks::var-mul {} (bv {} {}))))".format(
                        subname, self.get_elemwise_op(0), x_elems[i], self.operands[1], self.operands[0].bitwidth))
                self.partial_elems.append(subname)
        else:
            assert len(self.operands) == 3
            for i in range(m):
                for j in range(n):
                    subname = gen_name()
                    if is_custom_op(self.get_elemwise_op(0)):
                        self.partial_stmts.append(
                            "(define {} ({} (nks::var-add (nks::var-mul {} (bv {} {})) {}) #:len {}))".format(
                            subname, self.get_elemwise_op(0), x_elems[(i * n) + j], self.operands[2], \
                            self.operands[0].bitwidth, bias_elems[i], self.operands[0].bitwidth))
                    else:
                        self.partial_stmts.append(
                            "(define {} ({} (nks::var-add (nks::var-mul {} (bv {} {})) {})))".format(
                            subname, self.get_elemwise_op(0), x_elems[(i * n) + j], self.operands[2], \
                            self.operands[0].bitwidth, bias_elems[i]))
                    self.partial_elems.append(subname)

    def constituent_ops(self):
        return {self.get_elemwise_op(0), "nks::var-add"}
    
    def reduction_ops(self):
        return {"nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.activation {} {} ".format(
            self.name, self.get_elemwise_op(0), self.operands[0].name)
        if len(self.operands) == 2:
            if self.operands[1] != 1:
                string += "#:scale (bv {} {}) ".format(self.operands[1].name, self.operands[0].bitwidth)
        else:
            string += "#:bias {} ".format(self.operands[1].name)
            if self.operands[2] != 1:
                string += "#:scale (bv {} {}) ".format(self.operands[2].name, self.operands[0].bitwidth)
        string += "#:prec {}))".format(self.operands[0].bitwidth)
        return string


class NKI_tensor_reduce(TensorOp):
    def __init__(self, prefix, name, ast_expr, ops, data, axis, negate=False):
        super().__init__(prefix, name, ast_expr, "NKI_tensor_reduce", [data, axis, negate], ops)

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                elem_name = gen_name()
                string = "(define {} ({}{}))".format(
                    elem_name, self.get_elemwise_op(0), "".join(" {}".format(x_elems[(i * n) + j]) for j in range(n)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} ({}{}))".format(
                    elem_name, self.get_elemwise_op(0), "".join(" {}".format(x_elems[(i * n) + j]) for i in range(m)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {self.get_elemwise_op(0)}
    
    def reduction_ops(self):
        return {self.get_elemwise_op(0)}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.tensor_reduce {} {} #:axis {} ".format(
            self.name, self.get_elemwise_op(0), self.operands[0].name, self.operands[1])
        if self.operands[2]:
            string += "#:negate #t "
        string += "#:prec {}))".format(self.operands[0].bitwidth)
        return string


class NKI_tensor_partition_reduce(TensorOp):
    def __init__(self, prefix, name, ast_expr, ops, data):
        super().__init__(prefix, name, ast_expr, "NKI_tensor_partition_reduce", [data], ops)

    def generate_partial_ops(self, x_elems, m, n):
        for j in range(n):
            elem_name = gen_name()
            string = "(define {} ({}{}))".format(
                elem_name, self.get_elemwise_op(0), "".join(" {}".format(x_elems[(i * n) + j]) for i in range(m)))
            self.partial_stmts.append(string)
            self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {self.get_elemwise_op(0)}
    
    def reduction_ops(self):
        return {self.get_elemwise_op(0)}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.tensor_partition_reduce {} {} #:prec {}))".format(
            self.name, self.get_elemwise_op(0), self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_tensor_tensor(TensorOp):
    def __init__(self, prefix, name, ast_expr, data1, data2, ops):
        super().__init__(prefix, name, ast_expr, "NKI_tensor_tensor", [data1, data2], ops)
    
    def generate_partial_ops(self, x_elems, y_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} ({} {} {}))".format(
                name, self.get_elemwise_op(0), x_elems[i], y_elems[i])
            self.partial_stmts.append(string)
            self.partial_elems.append(name)
            
    def constituent_ops(self):
        return {self.get_elemwise_op(0)}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.tensor_tensor {} {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.get_elemwise_op(0), self.operands[0].bitwidth)
        return string


class NKI_reciprocal(TensorOp): 
    def __init__(self, prefix, name, ast_expr, data):
        super().__init__(prefix, name, ast_expr, "NKI_reciprocal", [data])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-reciprocal {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-reciprocal"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.isa.reciprocal {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_add(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_add", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-add {} {}))".format(
                    name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            # m1 = m2, but y_elems is broadcastable like numpy.add
            # x_elems -> (m1, n1) and y_elems -> (m1, n2)
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-add {} {}))".format(
                        name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-add"}
    
    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.add {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_subtract(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_subtract", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-sub {} {}))".format(
                    name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            # m1 = m2, but y_elems is broadcastable like numpy.add
            # x_elems -> (m1, n1) and y_elems -> (m1, n2)
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-sub {} {}))".format(
                        name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-sub"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.subtract {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_multiply(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_multiply", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-mul {} {}))".format(
                    name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            # m1 = m2, but y_elems is broadcastable like numpy.add
            # x_elems -> (m1, n1) and y_elems -> (m1, n2)
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-mul {} {}))".format(
                        name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-mul"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.multiply {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_divide(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_divide", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-div {} {}))".format(
                    name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            # m1 = m2, but y_elems is broadcastable like numpy.add
            # x_elems -> (m1, n1) and y_elems -> (m1, n2)
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-div {} {}))".format(
                        name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-div"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.divide {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_maximum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_maximum", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-max {} {}))".format(name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-max {} {}))".format(name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-max"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.maximum {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_minimum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "NKI_minimum", [x, y])
    
    def generate_partial_ops(self, x_elems, m1, n1, y_elems, m2, n2):
        if m1 == m2 and n1 == n2:
            for i in range(m1 * n1):
                name = gen_name()
                string = "(define {} (nks::var-min {} {}))".format(name, x_elems[i], y_elems[i])
                self.partial_stmts.append(string)
                self.partial_elems.append(name)
        else:
            assert m1 == m2 and n2 == 1 and n2 < n1
            for i in range(m1):
                for j in range(n1):
                    name = gen_name()
                    string = "(define {} (nks::var-min {} {}))".format(name, x_elems[(i * n1) + j], y_elems[i])
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-min"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.minimum {} {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1].name, self.operands[0].bitwidth)
        return string


class NKI_max(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "NKI_max", [x, axis])

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                elem_name = gen_name()
                string = "(define {} (nks::var-max {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} (nks::var-max {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for i in range(m)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-max"}
    
    def reduction_ops(self):
        return {"nks::var-max"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.max {} #:axis {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1], self.operands[0].bitwidth)
        return string


class NKI_min(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "NKI_min", [x, axis])

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                elem_name = gen_name()
                string = "(define {} (nks::var-min {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} (nks::var-min {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for i in range(m)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-min"}
    
    def reduction_ops(self):
        return {"nks::var-min"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.min {} #:axis {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1], self.operands[0].bitwidth)
        return string


class NKI_mean(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "NKI_mean", [x, axis])

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                sum = gen_name()
                string = "(define {} (nks::var-add {}))".format(
                    sum, " ".join(x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                elem_name = gen_name()
                string = "(define {} (nks::var-div {} (bv {} {})))".format(
                    elem_name, sum, n, self.operands[0].bitwidth)
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                sum = gen_name()
                string = "(define {} (nks::var-add {}))".format(
                    sum, " ".join(x_elems[(i * n) + j] for i in range(m)))
                self.partial_stmts.append(string)
                elem_name = gen_name()
                string = "(define {} (nks::var-div {} (bv {} {})))".format(
                    elem_name, sum, m, self.operands[0].bitwidth)
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-add", "nks::var-div"}
    
    def reduction_ops(self):
        return {"nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.mean {} #:axis {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1], self.operands[0].bitwidth)
        return string


class NKI_sum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "NKI_sum", [x, axis])

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                elem_name = gen_name()
                string = "(define {} (nks::var-add {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} (nks::var-add {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for i in range(m)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-add"}
    
    def reduction_ops(self):
        return {"nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.sum {} #:axis {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1], self.operands[0].bitwidth)
        return string


class NKI_prod(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "NKI_prod", [x, axis])

    def generate_partial_ops(self, x_elems, m, n):
        if self.operands[1] == 1:
            for i in range(m):
                elem_name = gen_name()
                string = "(define {} (nks::var-mul {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)
        else:
            for j in range(n):
                elem_name = gen_name()
                string = "(define {} (nks::var-mul {}))".format(
                    elem_name, " ".join(x_elems[(i * n) + j] for i in range(m)))
                self.partial_stmts.append(string)
                self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-mul"}
    
    def reduction_ops(self):
        return {"nks::var-mul"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.prod {} #:axis {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[1], self.operands[0].bitwidth)
        return string


class NKI_negative(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_negative", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-neg {}))".format(name, x_elems[i])
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-neg"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.negative {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_exp(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_exp", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-exp {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-exp"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.exp {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_log(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_log", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-log {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-log"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.log {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_sin(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_sin", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-sin {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-sin"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.sin {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_cos(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_cos", [x])

    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-cos {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-cos"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.cos {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_tan(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_tan", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-tan {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-tan"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.tan {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_tanh(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_tanh", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-tanh {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-tanh"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.tanh {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_arctan(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_arctan", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-arctan {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-arctan"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.arctan {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_sqrt(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_sqrt", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-sqrt {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-sqrt"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.sqrt {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_rsqrt(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_rsqrt", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-rsqrt {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-rsqrt"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.rsqrt {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_relu(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_relu", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-relu {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-relu"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.relu {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_softplus(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_softplus", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-softplus {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-softplus"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.softplus {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_mish(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_mish", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-mish {} #:len {}))".format(
                name, x_elems[i], self.operands[0].bitwidth)
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-mish"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.mish {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_square(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_square", [x])
    
    def generate_partial_ops(self, x_elems, m, n):
        for i in range(m * n):
            name = gen_name()
            string = "(define {} (nks::var-mul {} {}))".format(name, x_elems[i], x_elems[i])
            self.partial_stmts.append(string)
            self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-mul"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.square {} #:prec {}))".format(
            self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string


class NKI_softmax(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, bias=None):
        if bias == None:
            super().__init__(prefix, name, ast_expr, "NKI_softmax", [x])
        else:
            super().__init__(prefix, name, ast_expr, "NKI_softmax", [x, bias])

    def generate_partial_ops(self, x_elems, m, n, bias_elems=None):
        if bias_elems == None:
            assert len(self.operands) == 1
            for i in range(m):
                max_elems = gen_name()
                string = "(define {} (nks::var-max{}))".format(
                    max_elems, "".join(" " + x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                intermediate_elems = list()
                for j in range(n):
                    name = gen_name()
                    string = "(define {} (nks::var-exp (nks::var-sub {} {})) #:len {})".format(
                        name, x_elems[(i * n) + j], max_elems, self.operands[0].bitwidth)
                    self.partial_stmts.append(string)
                    intermediate_elems.append(name)
                sum_elems = gen_name()
                string = "(define {} (nks::var-add{}))".format(
                    sum_elems, "".join(" " + x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                for intermediate_elem in intermediate_elems:
                    name = gen_name()
                    string = "(define {} (nks::var-div {} {}))".format(name, intermediate_elem, sum_elems)
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)
        else:
            assert len(self.operands) == 2
            for i in range(m):
                max_elems = gen_name()
                string = "(define {} (nks::var-max{}))".format(
                    max_elems, "".join(" " + x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                intermediate_elems = list()
                for j in range(n):
                    name = gen_name()
                    string = "(define {} (nks::var-exp (nks::var-add (nks::var-sub {} {}) {})) #:len {})".format(
                        name, x_elems[(i * n) + j], max_elems, bias_elems[j], self.operands[0].bitwidth)
                    self.partial_stmts.append(string)
                    intermediate_elems.append(name)
                sum_elems = gen_name()
                string = "(define {} (nks::var-add{}))".format(
                    sum_elems, "".join(" " + x_elems[(i * n) + j] for j in range(n)))
                self.partial_stmts.append(string)
                for intermediate_elem in intermediate_elems:
                    name = gen_name()
                    string = "(define {} (nks::var-div {} {}))".format(name, intermediate_elem, sum_elems)
                    self.partial_stmts.append(string)
                    self.partial_elems.append(name)

    def constituent_ops(self):
        return {"nks::var-exp", "nks::var-max", "nks::var-add", "nks::var-reciprocal"}
    
    def reduction_ops(self):
        return {"nks::var-max", "nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.softmax {} ".format(self.name, self.operands[0].name)
        if len(self.operands) == 2:
            string += "#:bias {} ".format(self.operands[1].name)
        string += "#:prec {}))".format(self.operands[0].bitwidth)
        return string


class NKI_matmul(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y, transpose_x):
        super().__init__(prefix, name, ast_expr, "NKI_matmul", [x, y, transpose_x])

    def generate_partial_ops(self, x_elems, y_elems, m, n, k):
        if self.operands[2] == False:
            for i in range(m):
                for j in range(n):
                    elem_name = gen_name()
                    string = "(define {} (nks::var-add{}))".format(
                        elem_name,
                        "".join(" (nks::var-mul {} {})".format(
                        x_elems[(i * k) + o], y_elems[(o * n) + j]) for o in range(k))
                    )
                    self.partial_stmts.append(string)
                    self.partial_elems.append(elem_name)
        else:
            for i in range(m):
                for j in range(n):
                    elem_name = gen_name()
                    string = "(define {} (nks::var-add{}))".format(
                        elem_name,
                        "".join(" (nks::var-mul {} {})".format(
                        x_elems[(o * m) + i], y_elems[(o * n) + j]) for o in range(k))
                    )
                    self.partial_stmts.append(string)
                    self.partial_elems.append(elem_name)

    def constituent_ops(self):
        return {"nks::var-mul", "nks::var-add"}
    
    def reduction_ops(self):
        return {"nks::var-add"}

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.matmul {} {} ".format(
                    self.name, self.operands[0].name, self.operands[1].name)
        string += "#:transpose_x? {} ".format(str(self.operands[2]))
        string += "#:prec {}))".format(self.operands[0].bitwidth)
        return string


class NKI_transpose(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "NKI_transpose", [x])

    def generate_partial_ops(self, x_elems, m, n):
        for j in range(n):
            for i in range(m):
                self.partial_elems.append(x_elems[(i * n) + j])

    def to_rosette(self, space=0):
        string = " " * space + "(define {} (nki.lang.transpose {} #:prec {}))".format(
                self.name, self.operands[0].name, self.operands[0].bitwidth)
        return string

