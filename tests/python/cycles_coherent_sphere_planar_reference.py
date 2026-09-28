#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Double oracle for one sphere R/TT bracketed by separate planar mirrors.

Imports only independent scalar sphere references, never renderer utilities.
Finite mirror extents/global scene visibility remain separate renderer checks.
"""
import json
import numpy as np
from cycles_coherent_sphere_reference import sphere_path, spreading_fd
from cycles_coherent_sphere_transmission_reference import inventory, spreading, basis


def reflect(point, center, normal):
    return point - 2 * np.dot(point-center, normal) * normal


def intersection(a, b, center, normal):
    t = np.dot(center-a, normal) / np.dot(b-a, normal)
    if not 0 < t < 1:
        raise ValueError('mirror lies outside unfolded segment')
    return a + t * (b-a)


def fixture(kind):
    s = np.array([-.5, .1, .04])
    r = np.array([-.6, -.2, -.03]) if kind == 'R' else np.array([.4, .015, -.03])
    p = np.array([-1., 0., 0.]); q = np.array([-.9 if kind == 'R' else 1., 0., 0.])
    n = np.array([1., 0., 0.]); c = np.zeros(3); radius = .3; ior = 1.5
    vs = reflect(s, p, n); vr = reflect(r, q, n)
    branches = [sphere_path(vs, vr, c, radius)] if kind == 'R' else inventory(vs, vr, c, radius, ior)
    result = []
    for branch in branches:
        curved = [branch['point']] if kind == 'R' else branch['points']
        points = [intersection(vs, curved[0], p, n), *curved,
                  intersection(curved[-1], vr, q, n)]
        vertices = [s, *points, r]
        directions = [(b-a)/np.linalg.norm(b-a) for a,b in zip(vertices,vertices[1:])]
        lengths = [np.linalg.norm(b-a) for a,b in zip(vertices,vertices[1:])]
        weights = [1.] * len(lengths)
        if kind == 'TT': weights[2] = ior
        opl = float(np.dot(weights,lengths))
        expected_opl = float(branch['optical_length'] if kind == 'R' else branch['opl'])
        residuals = []
        for index in (0,len(points)-1):
            residuals.append(float(np.linalg.norm(directions[index]-2*np.dot(directions[index],n)*n-directions[index+1])))
        normals = [branch['normal']] if kind == 'R' else branch['normals']
        for j, normal in enumerate(normals,1):
            grad = weights[j]*directions[j]-weights[j+1]*directions[j+1]
            residuals.append(float(np.linalg.norm(basis(normal).T@grad)))
        receiver_normal = np.array([1.,0.,0.])
        virtual_normal = reflect(receiver_normal,np.zeros(3),n)
        if kind == 'R':
            detector = basis(virtual_normal)
            spread = float(spreading_fd(vs,vr,c,radius,detector[:,0],detector[:,1],1e-5))
        else: spread = float(spreading(branch,virtual_normal))
        assert abs(opl-expected_opl)<2e-14 and max(residuals)<2e-13
        result.append({'source':s.tolist(),'receiver':r.tolist(),'prefix_plane':p.tolist(),
                       'suffix_plane':q.tolist(),'sphere_radius':radius,'ior':ior,
                       'points':[x.tolist() for x in points], 'opl':opl,'spreading':spread,
                       'morse':int(branch.get('morse',0)), 'max_law_residual':max(residuals),
                       'unfolded_opl_error':abs(opl-expected_opl)})
    return result

if __name__ == '__main__':
    print(json.dumps({kind:fixture(kind) for kind in ('R','TT')},indent=2))
