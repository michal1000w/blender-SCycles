#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent test-only closed oriented convex mesh validator examples."""
import numpy as np

def valid(v,f):
 edges={}
 for tri in f:
  for a,b in zip(tri,tri[1:]+tri[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
 if any(len(e)!=2 or e[0]!=e[1][::-1] for e in edges.values()):return False
 volume=sum(np.dot(v[a],np.cross(v[b],v[c])) for a,b,c in f)/6
 if abs(volume)<1e-12:return False
 for a,b,c in f:
  n=np.cross(v[b]-v[a],v[c]-v[a])*np.sign(volume)
  if np.linalg.norm(n)<1e-12 or np.max((v-v[a])@n)>1e-10:return False
 return True

if __name__=='__main__':
 v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
 f=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
 assert valid(v,f);assert valid(v,[t[::-1] for t in f]);assert not valid(v,f[:-1]);assert not valid(v,[f[0][::-1]]+f[1:]);assert not valid(v,f[:2]);concave=v.copy();concave[6]=[0,0,0];assert not valid(concave,f);print('PASS6 independent closed/convex/orientation cases')
