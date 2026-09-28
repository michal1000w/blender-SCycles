#!/usr/bin/env python3
"""Independent float64 isotropic GGX transmission evaluation for logged float failures."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for line in a.input.read_text().splitlines():
 if not line.startswith('flat_failure '):continue
 tokens=line.split()[1:];d={};i=0
 for key,n in [('wi',3),('wo',3),('ni',1),('no',1),('alpha',1),('ref_f0',1),('f0',3),('tint',3),('value',3),('native',3)]:
  assert tokens[i]==key;i+=1;v=list(map(float,tokens[i:i+n]));i+=n;d[key]=v if n>1 else v[0]
 wi,wo,ni,no=d['wi'],d['wo'],d['ni'],d['no'];assert wo[2]<0
 dot=lambda x,y:sum(u*v for u,v in zip(x,y))
 momentum=[ni*x+no*y for x,y in zip(wi,wo)];length2=dot(momentum,momentum)
 h=[v/math.sqrt(length2) for v in momentum]
 if h[2]<0:h=[-v for v in h]
 ci=dot(wi,h);co=-dot(wo,h);alpha2=d['alpha']**2
 D=alpha2/(math.pi*(h[0]**2+h[1]**2+alpha2*h[2]**2)**2)
 lam=lambda w:(math.sqrt(1+alpha2*(w[0]**2+w[1]**2)/w[2]**2)-1)/2
 rs=(ni*ci-no*co)/(ni*ci+no*co);rp=(no*ci-ni*co)/(no*ci+ni*co)
 F=(rs*rs+rp*rp)/2;s=min(1,max(0,(F-d['ref_f0'])/(1-d['ref_f0'])))
 common=D/wi[2]*no*no*ci*co/length2/(1+lam(wi)+lam(wo))
 reference=[(1-f0)*(1-s)*t*common for f0,t in zip(d['f0'],d['tint'])]
 error=lambda values:max(abs(x-y) for x,y in zip(values,reference))/max(1,max(reference))
 rows.append({'inputs':d,'double_reference':reference,'grating_error':error(d['value']),'native_error':error(d['native'])})
assert rows
out={'scope':'Float64 reference at the same logged float inputs; diagnostic cases only, not an exhaustive oracle', 'cases':rows,'maximum_grating_error':max(r['grating_error'] for r in rows),'maximum_native_error':max(r['native_error'] for r in rows)}
a.output.write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='cases'})
