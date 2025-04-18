"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Definitions of Python decorators for NKS.

"""


import ast
import inspect

from nks_parser import *
from synthesizer import *
from nki_codegen import *


# Define the synthesizer generator
Synthesizer = NKS_Synthesizer()


# This is a decorator that indicates that a function is an 
# algorithmic specification of a NKI kernel.
def spec(func):
    # Parse the source code into an AST
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    # Generate semantics for the original specification
    semagen = SpecSemaGenerator(tree)
    Synthesizer.org_spec = semagen.statements
    # Record function name, function arguments, etc.
    if Synthesizer.func_name:
        assert Synthesizer.func_name == semagen.func_name
    else:
        Synthesizer.func_name = semagen.func_name
    for idx, arg in enumerate(semagen.func_args):
        if len(Synthesizer.func_args) >= idx + 1:
            assert Synthesizer.func_args[idx] == semagen.val_defs[arg]
        else:
            Synthesizer.func_args.append(semagen.val_defs[arg])
    return func


# This is a decorator that indicates that a function contains a
# sketch of a NKI kernel.
def sketch(func):
    # Parse the source code into an AST
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    # Generate the loopless sketch
    semagen = LoopLessSemaGenerator(tree)
    Synthesizer.org_loopless_sketch = semagen.statements
    # Perform synthesis on the loopless sketch
    Synthesizer.generate_synthesizer(num_elems=2)

    # Generate the full sketch
    org_semagen = OrgSemaGenerator(tree, Synthesizer.hole_defs, semagen.replacement_vals)
    org_semagen.statements = org_semagen.assignments + org_semagen.statements
    Synthesizer.org_sketch = org_semagen.statements

    # Perform synthesis on the full sketch
    Synthesizer.generate_full_synthesizer()
    NKICodegen(Synthesizer, tree)
    os._exit(0)
    
    return func
