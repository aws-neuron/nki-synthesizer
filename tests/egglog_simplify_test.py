"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Egglog test for simplification of expressions
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


a, b, c = vars_("a b c", Num)

rule_set = ruleset(
        rewrite(a / b).to(c),
        rewrite(a * b).to(b * a),
        rewrite(a * (b * c)).to((a * b) * c),
        rewrite(a * (b / c)).to((a * b) / c),
        rewrite((a / b) * c).to(a * (c / b)),
        rewrite((a / b) / c).to(a / (b * c)),
        rewrite((a * b) * c).to(a * (b * c)),
        rewrite((a * b) / c).to(a * (b / c)),
        rewrite(a / (b / c)).to((a / b) * c),
        rewrite(a / (b * c)).to((a / b) / c),
        rewrite(a / Num(1)).to(a),
        rewrite(a * Num(1)).to(a),
        rewrite(a / a).to(Num(1)),
        rewrite(a * Num(0)).to(Num(0)),
        rewrite(Num(0) / a).to(Num(0))
    )


expr1 = a / b
expr2 = (Num.var("x") * Num.var("y")) / (Num.var("x"))
expr3 =(Num.var("x") * Num.var("y")) / (Num.var("z") * Num.var("x"))

print(egraph.simplify(expr1,  rule_set * 10))
print(egraph.simplify(expr2, rule_set * 10))

simplified_expr3 = egraph.simplify(expr3, rule_set * 10)
print(simplified_expr3)
print(type(simplified_expr3))

new_expr = simplified_expr3 / Num.var("y")

print(egraph.simplify(new_expr, rule_set * 10))
