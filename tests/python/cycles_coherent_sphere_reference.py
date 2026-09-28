#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent double scalar-angle convex-sphere reflection and ray-map oracle.

Uses a monotone one-dimensional angular root in the source/center/receiver
plane. No renderer geometry/field utility is imported. Exterior convex unit
mirror only; a source/receiver horizon intersection is an explicit rejection.
"""
import json
import math
from pathlib import Path
import sys
import numpy as np


def unit(v):
    return v / np.linalg.norm(v, axis=-1)[..., None]


def sphere_path(source, receiver, center, radius):
    source, receiver, center = (np.asarray(p, dtype=np.float64) for p in (source, receiver, center))
    a, b = source-center, receiver-center
    da, db = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if radius <= 0 or np.any(da <= radius) or np.any(db <= radius):
        raise ValueError('exterior positive-radius sphere required')
    e1, er = a / da[..., None], b / db[..., None]
    beta = np.arccos(np.clip(np.sum(e1 * er, axis=-1), -1.0, 1.0))
    orth = er - np.sum(e1*er, axis=-1)[..., None]*e1
    orthlen = np.linalg.norm(orth, axis=-1)
    fallback = np.cross(e1, np.array([0.,0.,1.]))
    if np.linalg.norm(fallback) < 1e-12: fallback = np.cross(e1,np.array([0.,1.,0.]))
    e2 = np.where((orthlen > 1e-12)[..., None], orth / np.maximum(orthlen,1e-300)[..., None], unit(fallback))
    lo = np.maximum(0., beta-np.arccos(radius/db))
    hi = np.minimum(beta, np.arccos(radius/da))
    if np.any(lo > hi): raise ValueError('no jointly visible sphere cap')
    for _ in range(60):
        alpha = (lo+hi)/2
        la=np.sqrt(da*da+radius*radius-2*da*radius*np.cos(alpha))
        lb=np.sqrt(db*db+radius*radius-2*db*radius*np.cos(beta-alpha))
        derivative=da*np.sin(alpha)/la-db*np.sin(beta-alpha)/lb
        lo=np.where(derivative<0,alpha,lo);hi=np.where(derivative>=0,alpha,hi)
    alpha=(lo+hi)/2
    normal=np.cos(alpha)[...,None]*e1+np.sin(alpha)[...,None]*e2
    hit=center+radius*normal
    incident=unit(hit-source);outgoing=unit(receiver-hit)
    la=np.linalg.norm(hit-source,axis=-1);lb=np.linalg.norm(receiver-hit,axis=-1)
    if np.any(np.sum((source-hit)*normal,axis=-1) <= 0) or np.any(np.sum((receiver-hit)*normal,axis=-1) <= 0):
        raise ValueError('stationary branch is not visible exterior reflection')
    return {'point':hit,'normal':normal,'incident':incident,'outgoing':outgoing,
            'source_distance':la,'receiver_distance':lb,'optical_length':la+lb}


def spreading(path, detector_u, detector_v, radius):
    # Tangent Hessian includes the real sphere curvature 2*cos(i)/r.
    n,wa,wb=(path[k] for k in ('normal','incident','outgoing'))
    tangent=unit(np.cross(n,np.array([0.,0.,1.])))
    second=np.cross(n,tangent);basis=np.stack((tangent,second),axis=-1)
    identity=np.eye(3)
    A=identity-wa[..., :,None]*wa[...,None,:]
    B=identity-wb[..., :,None]*wb[...,None,:]
    la,lb=path['source_distance'],path['receiver_distance']
    cart=A/la[...,None,None]+B/lb[...,None,None]
    H=np.swapaxes(basis,-2,-1)@cart@basis
    H+=(-2*np.sum(wa*n,axis=-1)/radius)[...,None,None]*np.eye(2)
    directions=np.stack((detector_u,detector_v),axis=-1)
    rhs=np.swapaxes(basis,-2,-1)@B@directions/lb[...,None,None]
    differential=A@basis@np.linalg.solve(H,rhs)/la[...,None,None]
    return np.abs(np.sum(wa*np.cross(differential[...,0],differential[...,1]),axis=-1))


def spreading_fd(source, receiver, center, radius, detector_u, detector_v, step):
    deriv=[]
    for direction in (detector_u,detector_v):
        plus=sphere_path(source,receiver+step*direction,center,radius)['incident']
        minus=sphere_path(source,receiver-step*direction,center,radius)['incident']
        deriv.append((plus-minus)/(2*step))
    incident=sphere_path(source,receiver,center,radius)['incident']
    return np.abs(np.sum(incident*np.cross(*deriv),axis=-1))


def modes(path):
    incident,n=path['incident'],path['normal']
    projected=(np.eye(3)-incident[..., :,None]*incident[...,None,:])/math.sqrt(2.)
    householder=np.eye(3)-2*n[..., :,None]*n[...,None,:]
    return householder@projected


def checks():
    errors=[];axis_errors=[];power_errors=[]
    u,v=np.array([0.,1.,0.]),np.array([0.,0.,1.])
    for scale in (.01,1.,100.):
        source=scale*np.array([-1.5,.1,.4]);receiver=scale*np.array([-1.2,-.2,-.3]);r=.3*scale
        p=sphere_path(source,receiver,np.zeros(3),r)
        analytic=spreading(p,u,v,r);fd=spreading_fd(source,receiver,np.zeros(3),r,u,v,1e-5*scale)
        errors.append(float(abs(analytic-fd)/analytic))
        reflected=p['incident']-2*np.dot(p['incident'],p['normal'])*p['normal']
        assert np.max(abs(reflected-p['outgoing']))<2e-14
        power_errors.append(float(abs(np.sum(modes(p)**2)-1)))
        s=scale*np.array([-1.5,0.,0.]);t=scale*np.array([-1.2,0.,0.]);p=sphere_path(s,t,np.zeros(3),r)
        reference=1/(1.2*scale+.9*scale+2*1.2*.9*scale*scale/r)**2
        axis_errors.append(float(abs(spreading(p,u,v,r)-reference)/reference))
    assert max(errors)<1e-7 and max(axis_errors)<1e-13 and max(power_errors)<1e-14
    for source,receiver in (([.1,0,0],[-1,0,0]),([1,0,0],[-1,0,0])):
        try:sphere_path(source,receiver,[0,0,0],.3)
        except ValueError:pass
        else:raise AssertionError('interior/hidden branch accepted')
    return {'finite_difference_spreading_max_relative':max(errors),'axial_closed_form_max_relative':max(axis_errors),'three_dipole_unit_total_power_error':max(power_errors),'reflection_law':'<2e-14','scale_checks':[.01,1,100],'interior_and_hidden_rejected':True}


if __name__=='__main__':
    result=checks();print(json.dumps(result,indent=2))
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
