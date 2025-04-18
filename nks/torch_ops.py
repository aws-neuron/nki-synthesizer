"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Translation layer between Python to Rosette and vice versa.

"""


from ir import *


class Torch_add(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_add", [x, y])
        
    def constituent_ops(self):
        return {"bvadd"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.add "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_subtract(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_subtract", [x, y])

    def constituent_ops(self):
        return {"bvsub"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.subtract "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_multiply(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_multiply", [x, y])

    def constituent_ops(self):
        return {"bvmul"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.multiply "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_divide(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_divide", [x, y])

    def constituent_ops(self):
        return {"bvsdiv"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.divide "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_maximum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_maximum", [x, y])

    def constituent_ops(self):
        return {"bvsmax"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.maximum "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_minimum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_minimum", [x, y])

    def constituent_ops(self):
        return {"bvsmin"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.minimum "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string
    

class Torch_max(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_max", [x, axis])

    def constituent_ops(self):
        return {"bvsmax"}
    
    def reduction_ops(self):
        return {"bvsmax"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.max "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string
    

class Torch_min(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_min", [x, axis])

    def constituent_ops(self):
        return {"bvsmin"}
    
    def reduction_ops(self):
        return {"bvsmin"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.min "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_mean(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_mean", [x, axis])

    def constituent_ops(self):
        return {"bvadd", "bvsdiv"}
    
    def reduction_ops(self):
        return {"bvadd"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.mean "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_sum(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_sum", [x, axis])

    def constituent_ops(self):
        return {"bvadd"}
    
    def reduction_ops(self):
        return {"bvadd"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.sum "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_prod(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_prod", [x, axis])

    def constituent_ops(self):
        return {"bvmul"}
    
    def reduction_ops(self):
        return {"bvmul"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.prod "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string
    

class Torch_negative(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_negative", [x])

    def constituent_ops(self):
        return {"bvneg"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.negative "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_exp(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_exp", [x])
    
    def constituent_ops(self):
        return {"bvexp"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.exp "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_log(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_log", [x])

    def constituent_ops(self):
        return {"bvlog"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.log "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_sin(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_sin", [x])

    def constituent_ops(self):
        return {"bvsin"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.sin "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_cos(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_cos", [x])

    def constituent_ops(self):
        return {"bvcos"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.cos "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_tan(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_tan", [x])

    def constituent_ops(self):
        return {"bvtan"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.tan "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_tanh(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_tanh", [x])

    def constituent_ops(self):
        return {"bvtanh"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.tanh "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_arctan(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_arctan", [x])

    def constituent_ops(self):
        return {"bvarctan"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.arctan "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_sqrt(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_sqrt", [x])

    def constituent_ops(self):
        return {"bvsqrt"}
    
    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.sqrt "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_rsqrt(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_rsqrt", [x])

    def constituent_ops(self):
        return {"bvrsqrt"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.rsqrt "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_relu(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_relu", [x])

    def constituent_ops(self):
        return {"bvrelu"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.relu "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_softplus(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_softplus", [x])

    def constituent_ops(self):
        return {"bvsoftplus"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.softplus "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_mish(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_mish", [x])

    def constituent_ops(self):
        return {"bvmish"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.mish "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_square(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_square", [x])
        
    def constituent_ops(self):
        return {"bvmul"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.square "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_softmax(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, axis):
        super().__init__(prefix, name, ast_expr, "Torch_softmax", [x, axis])

    def constituent_ops(self):
        return {"bvexp", "bvsdiv", "bvsmax", "bvadd"}

    def reduction_ops(self):
        return {"bvsmax", "bvadd"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.softmax "
        string += self.operands[0].name + " "
        string += "#:axis " + str(self.operands[1]) + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_matmul(TensorOp):
    def __init__(self, prefix, name, ast_expr, x, y):
        super().__init__(prefix, name, ast_expr, "Torch_matmul", [x, y])

    def constituent_ops(self):  
        return {"bvadd", "bvmul"}

    def reduction_ops(self):
        return {"bvadd"}

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.matmul "
        string += self.operands[0].name + " "
        string += self.operands[1].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string


class Torch_transpose(TensorOp):
    def __init__(self, prefix, name, ast_expr, x):
        super().__init__(prefix, name, ast_expr, "Torch_transpose", [x])

    def to_rosette(self, space=0):
        string = ""
        for _ in range(space):
            string += " "
        string += "(define " + self.name + " "
        string += "(torch.transpose "
        string += self.operands[0].name + " "
        string +=  "#:prec " + str(self.operands[0].bitwidth) + "))"
        return string

