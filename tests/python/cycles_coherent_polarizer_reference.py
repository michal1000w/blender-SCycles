#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent ideal linear polarizer Jones/Malus and Fresnel glare oracle.

References: OpenStax University Physics3 §1.7 and Oxford Fresnel/Brewster lab.
No renderer functions, clamping, rendered pixel oracle or brightness fit.
"""
import json,math
import numpy as np
SOURCES=['https://openstax.org/books/university-physics-volume-3/pages/1-7-polarization','https://users.physics.ox.ac.uk/~lvovsky/471/labs/fresnel_brewster.pdf']


def unit(v):return np.asarray(v,dtype=float)/np.linalg.norm(v)
def projector(direction,axis):
    direction=unit(direction);axis=np.array(axis,dtype=float,copy=True);axis-=direction*np.dot(axis,direction)
    if np.linalg.norm(axis)<1e-12:raise ValueError('polarizer axis parallel to propagation')
    axis=unit(axis);return np.outer(axis,axis)
def unpolarized_modes(direction):
    direction=unit(direction);return (np.eye(3)-np.outer(direction,direction))/math.sqrt(2)
def power(modes):return float(np.vdot(modes,modes).real)
def axis(degrees):
    theta=math.radians(degrees);return np.array([math.cos(theta),math.sin(theta),0.])


def checks():
    direction=np.array([0.,0.,1.]);modes=unpolarized_modes(direction);first=modes@projector(direction,axis(0));malus=[]
    assert abs(power(modes)-1)<1e-14 and abs(power(first)-.5)<1e-14
    for theta in (0,15,30,45,60,75,90,180):
        actual=power(first@projector(direction,axis(theta)));expected=.5*math.cos(math.radians(theta))**2;assert abs(actual-expected)<1e-14
        assert np.max(abs(projector(direction,axis(theta))-projector(direction,-axis(theta))))<1e-14
        malus.append({'relative_angle_degrees':theta,'incident_fraction':actual,'closed_form_fraction':expected})
    bridge=first@projector(direction,axis(45))@projector(direction,axis(90));assert abs(power(bridge)-.125)<1e-14
    brewster=[]
    for index in (1.33,1.5):
        theta=math.atan(index);incoming=np.array([math.sin(theta),0.,-math.cos(theta)]);outgoing=np.array([math.sin(theta),0.,math.cos(theta)]);normal=np.array([0.,0.,1.]);s=unit(np.cross(incoming,normal));pi=np.cross(s,incoming);po=np.cross(s,outgoing)
        ci=math.cos(theta);ct=math.sqrt(1-math.sin(theta)**2/index**2);rs=(ci-index*ct)/(ci+index*ct);rp=(index*ci-ct)/(index*ci+ct);modes=unpolarized_modes(incoming);reflected=(modes@s)[:,None]*(rs*s)+(modes@pi)[:,None]*(rp*po)
        assert abs(rp)<1e-14
        passing_s=power(reflected@projector(outgoing,s));passing_p=power(reflected@projector(outgoing,po));passing45=power(reflected@projector(outgoing,unit(s+po)))
        assert passing_p<1e-28 and abs(passing_s-.5*rs*rs)<1e-14 and abs(passing45-.5*passing_s)<1e-14
        brewster.append({'ior':index,'brewster_degrees':math.degrees(theta),'unfiltered_glare_fraction':power(reflected),'s_axis_glare_fraction':passing_s,'p_axis_glare_fraction':passing_p,'45deg_glare_fraction':passing45})
    return {'source_model':'Unit unpolarized world-axis dipole ensemble; projector P=aa^T on the transverse pass axis','single_filter_fraction':power(first),'malus':malus,'crossed_with_45deg_middle_filter_fraction':power(bridge),'brewster':brewster,'primary_sources':SOURCES}

if __name__=='__main__':print(json.dumps(checks(),indent=2))
