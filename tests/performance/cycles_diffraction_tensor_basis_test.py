# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression for float overflow next to interpolation knots."""
import unittest
import numpy as np
import cycles_diffraction_tensor_fit as scalar
import cycles_diffraction_tensor_anchor as anchored

class TensorBasisTest(unittest.TestCase):
    def test_adjacent_knots(self):
        for implementation in (scalar,anchored):
            for dtype in (np.float32,np.float64):
                nodes=np.linspace(0,1,7).astype(dtype)
                weights=implementation.weights(nodes.astype(float),5).astype(dtype)
                with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                    for node in nodes:
                        for q in (np.nextafter(node,dtype(-np.inf)),node,np.nextafter(node,dtype(np.inf))):
                            values=implementation.basis(q,nodes,weights)
                            self.assertTrue(np.isfinite(values).all())
                            self.assertLess(abs(values.sum()-1),16*np.finfo(dtype).eps)
                            self.assertLess(abs(values@nodes-q),16*np.finfo(dtype).eps)

    def test_polynomial_reproduction(self):
        nodes=np.linspace(0,1,7)
        for implementation in (scalar,anchored):
            weights=implementation.weights(nodes,5)
            for degree in range(6):
                for x in np.linspace(0,1,257):
                    self.assertLess(abs(implementation.basis(x,nodes,weights)@nodes**degree-x**degree),1e-13)

if __name__=='__main__':unittest.main()
