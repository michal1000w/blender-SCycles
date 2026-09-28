#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Precision attribution, NOT a Metal implementation or performance benchmark.

Keep complex64 storage while optionally accumulating products/residuals in
complex128 on CPU. This determines whether compensated GPU arithmetic merits
implementation; native Metal double arithmetic is not assumed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np


class Arithmetic(ast.NodeTransformer):
    def __init__(self, fused=False):
        self.fused=fused

    def visit_BinOp(self,node):
        if self.fused and isinstance(node.op,(ast.Add,ast.Sub)):
            if isinstance(node.right,ast.BinOp) and isinstance(node.right.op,ast.MatMult):
                return ast.copy_location(ast.Call(func=ast.Name(id='diagnostic_matmul_add',ctx=ast.Load()),
                    args=[self.visit(node.right.left),self.visit(node.right.right),self.visit(node.left),
                          ast.Constant(-1 if isinstance(node.op,ast.Sub) else 1)],keywords=[]),node)
            if isinstance(node.op,ast.Add) and isinstance(node.left,ast.BinOp) and isinstance(node.left.op,ast.MatMult):
                return ast.copy_location(ast.Call(func=ast.Name(id='diagnostic_matmul_add',ctx=ast.Load()),
                    args=[self.visit(node.left.left),self.visit(node.left.right),self.visit(node.right),
                          ast.Constant(1)],keywords=[]),node)
        self.generic_visit(node)
        if isinstance(node.op,ast.MatMult):
            return ast.copy_location(ast.Call(func=ast.Name(id='diagnostic_matmul',ctx=ast.Load()),
                                             args=[node.left,node.right],keywords=[]),node)
        return node

    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and node.func.attr=='solve' and
            isinstance(node.func.value,ast.Attribute) and node.func.value.attr=='linalg'):
            node.func=ast.Name(id='diagnostic_solve',ctx=ast.Load())
        return node


def load(path,name,products,refinement):
    def matmul(a,b):
        dtype=np.result_type(a,b)
        if products in ('compensated','wide_compensated') and dtype==np.complex64:
            from cycles_diffraction_compensated_product import compensated_matmul
            return compensated_matmul(a,b)
        if products and dtype==np.complex64:
            return (a.astype(complex)@b.astype(complex)).astype(dtype)
        return a@b

    def matmul_add(a,b,c,sign):
        dtype=np.result_type(a,b,c)
        if dtype==np.complex64:
            return (c.astype(complex)+sign*(a.astype(complex)@b.astype(complex))).astype(dtype)
        return c+sign*(a@b)

    def solve(a,b):
        x=np.linalg.solve(a,b)
        if refinement and np.result_type(a,b)==np.complex64:
            # Residual in double, correction solve and stored update in float.
            for _ in range(2):
                residual=(b.astype(complex)-a.astype(complex)@x.astype(complex)).astype(np.complex64)
                x=x+np.linalg.solve(a,residual)
        return x

    module=types.ModuleType(name)
    module.__file__=str(path)
    module.diagnostic_matmul=matmul
    module.diagnostic_solve=solve
    module.diagnostic_matmul_add=matmul_add
    source=path.read_text()
    if products in ('wide_step','wide_compensated') and path.name=='cycles_diffraction_boundary_cascade.py':
        source=source.replace('norm/1.0','norm/8.0').replace('range(1, 21)','range(1, 41)')
    if products=='initial_double' and path.name=='cycles_diffraction_boundary_cascade.py':
        source=source.replace('p, q = p.astype(dtype), q.astype(dtype)',
                              'p, q = p.astype(np.complex128), q.astype(np.complex128)')
        source=source.replace('eye = np.eye(m, dtype=dtype)','eye = np.eye(m, dtype=np.complex128)')
        source=source.replace('term = np.eye(2*m, dtype=dtype)','term = np.eye(2*m, dtype=np.complex128)')
        source=source.replace('    regular = False',
                              '    a,b,c,d = (v.astype(dtype) for v in (a,b,c,d))\n    eye=eye.astype(dtype)\n    regular = False')
    tree=ast.fix_missing_locations(Arithmetic(products == "fused").visit(ast.parse(source)))
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compensated',action='store_true')
    parser.add_argument('--fused',action='store_true')
    parser.add_argument('--initial-double',action='store_true')
    parser.add_argument('--wide-step',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    here=Path(__file__).resolve().parent
    corepath=here/'cycles_diffraction_boundary_cascade.py'
    driverpath=here/'cycles_diffraction_boundary_basis.py'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),corepath,driverpath)}
    summary={}
    treatments=([('wide_compensated','wide_compensated',False)] if args.wide_step and args.compensated else [('wide_step','wide_step',False)] if args.wide_step else [('initial_double','initial_double',False)] if args.initial_double else [('fused','fused',False)] if args.fused else [('compensated','compensated',False)] if args.compensated else
                [('baseline',False,False),('products',True,False),('solves',False,True),('both',True,True)])
    if args.compensated:
        extra=here/'cycles_diffraction_compensated_product.py'
        hashes[str(extra)]=hashlib.sha256(extra.read_bytes()).hexdigest()
    for name,products,refinement in treatments:
        core=load(corepath,'cycles_diffraction_boundary_cascade',products,refinement)
        sys.modules['cycles_diffraction_boundary_cascade']=core
        driver=load(driverpath,'precision_driver',products,refinement)
        output=args.output/(name+'.json')
        sys.argv=[str(driverpath),'--output',str(output)]
        try:
            driver.main()
        except SystemExit as result:
            if result.code not in (0,1): raise
        data=json.loads(output.read_text())
        data['instrumentation']=dict(product_accumulation=products,
                                     solves_double_residual_refinement=refinement,
                                     source_hashes=hashes,scope=__doc__)
        output.write_text(json.dumps(data,indent=2)+'\n')
        summary[name]=data['summary']
    (args.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,sources=hashes,
                                                          treatments=summary),indent=2)+'\n')


if __name__=='__main__': main()
