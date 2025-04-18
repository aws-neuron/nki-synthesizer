"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Egglog test for checking equality of expressions
"""

from __future__ import annotations

from egglog import *


egraph = EGraph()

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...
    
    def __truediv__(self, other: Num) -> Num: ...
    
    def __floordiv__(self, other: Num) -> Num: ...


dim1 = Num.var("dim1")
dim2 = Num.var("dim2")

expr1 = dim1 / Num.var("m")
expr2 = dim2 / Num.var("m")

print(type(expr1))
print(type(expr2))

egraph.register(expr1, expr2)

rule_set = ruleset(
        rewrite(dim1).to(dim2),
        rewrite(dim2).to(dim1)
    )

egraph.run(rule_set * 10)

print(egraph.check(expr1 == expr2))