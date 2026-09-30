"""Independent equations, root solver, Jacobian differences, and time integration."""
from pathlib import Path
import json, hashlib
import numpy as np
import scipy
from scipy.special import expit
from scipy.optimize import root
from scipy.integrate import solve_ivp

out=Path(__file__).parent
base=dict(a=17.,b=13.,c=19.,d=6.,h=8.,r=.2)
def balance(x,p,B):
 e,i=x;u=B[0]+p['a']*e-p['b']*i;v=B[1]+p['c']*e-p['d']*i
 fe=expit(5*(u-1.5));fi=expit(5*(v-4))*expit(-5*(v-p['h']))*(-np.expm1(-5*(p['h']-4)))
 return np.array([-e+(1-e)*fe,-i+(1-i)*fi])
def jac(x,p,B):
 h=1e-6
 return np.column_stack([(balance(x+np.eye(2)[k]*h,p,B)-balance(x-np.eye(2)[k]*h,p,B))/(2*h) for k in range(2)])
def roots(p,B):
 found=[]
 for e in np.linspace(0,.5,41):
  for i in np.linspace(0,.5,41):
   x=root(lambda x:balance(x,p,B),[e,i],tol=1e-11).x
   if min(x)>=-1e-10 and max(x)<=.5+1e-10 and max(abs(balance(x,p,B)))<1e-9 and all(max(abs(x-y))>1e-7 for y in found):found.append(x)
 return sorted(found,key=lambda x:x[0])
def evolve(x,p,B):
 s=solve_ivp(lambda t,x:balance(x,p,B)/[7.8,7.8*p['r']],(0.,10000.),x,method='DOP853',atol=1e-12,rtol=1e-12,t_eval=[9800.,9900.,10000.])
 assert s.success
 assert max(abs(balance(s.y[:,-1],p,B)))<1e-8
 assert max(np.ptp(s.y,axis=1))<1e-6
 return s.y[:,-1]
results=[]
rs=roots(base,[.35,0.]);stable=[x for x in rs if max(np.linalg.eigvals(jac(x,base,[.35,0.])/[ [7.8],[7.8*.2] ]).real)<0]
assert len(rs)==7 and len(stable)==4
H=min([x for x in stable if x[0]>.45],key=lambda x:-x[1]);S=min([x for x in stable if x[0]>.45],key=lambda x:x[1])
for role,x,expected in [('herald',H,[.295856318434,.284152067523]),('seizure',S,[.5,.000561853889])]:
 y=evolve(x,base,[.1,0.]);assert max(abs(y-expected))<1e-7
 results.append(dict(protocol='positive_input_withdrawal',source=role,initial=x.tolist(),final=y.tolist()))
p=dict(base,a=16.);rs=roots(p,[.5,0.]);S=max(rs,key=lambda x:x[0]);y=evolve(S,dict(p,h=12.),[.5,0.]);assert max(abs(y-[.301651801454,.298554492072]))<1e-7
results.append(dict(protocol='raised_threshold',final=y.tolist()))
# Check both input derivatives at the active source against independent roots.
x=min([x for x in stable if .1<x[0]<.4],key=lambda x:x[0]);h=1e-5
susceptibility=np.column_stack([(root(lambda y:balance(y,base,np.array([.35,0.])+h*np.eye(2)[k]),x).x-root(lambda y:balance(y,base,np.array([.35,0.])-h*np.eye(2)[k]),x).x)/(2*h) for k in range(2)])
results.append(dict(protocol='finite_difference_susceptibility',gain=susceptibility.tolist()))
# I stimulation can push inhibition onto its failing tail; do not treat it as
# monotone suppression. These are independent equilibrium observations.
for B in ([.35,1.],[.35,8.],[.35,16.],[16.,16.]):
 rs=roots(base,B)
 results.append(dict(protocol='nonzero_I_roots',input=B,roots=[x.tolist() for x in rs]))
report=dict(scipy=scipy.__version__,checks=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(checks=len(results),selected_source_roots=7)))
