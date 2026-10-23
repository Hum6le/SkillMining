"""Small exact sanity checks for the proposed tree-free TraceCover proof.

This is a theory verifier, not a production mining implementation. It does not
modify training artifacts or call an LLM. Exact Fractions check the elementary
bound against exhaustive optima; SciPy checks the proposed LP numerically.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as Q
from itertools import combinations
import json
from pathlib import Path
import random


def coverage(selected, traces, rho):
    return sum((min(Q(1),Q(sum(c for e,c in row.items() if e in selected),1)/(rho*sum(row.values())))
                for row in traces),Q(0))/len(traces)


def cost(selected,p,gamma):
    deg=Counter(u for u,v in selected)
    return Q(len(selected))+gamma*sum((p.get(u,Q(0))*d*(d-1)/2 for u,d in deg.items()),Q(0))


def reachable(selected,nodes,entries):
    seen=set(entries)
    while True:
        enlarged=seen|{v for u,v in selected if u in seen}
        if enlarged==seen:return seen==set(nodes)
        seen=enlarged


def lower_bound(edges,nodes,entries,traces,rho,eta,p,gamma):
    scores=sorted((coverage({e},traces,rho) for e in edges),reverse=True)
    mass=Q(0);k_cov=None
    for k,v in enumerate(scores,1):
        mass+=v
        if mass>=eta:k_cov=k;break
    assert k_cov is not None
    k0=max(k_cov,len(nodes)-len(entries))
    capacities=Counter(u for u,v in edges)
    increments=sorted(p.get(u,Q(0))*j for u,d in capacities.items() for j in range(d))
    phi=sum(increments[:k0],Q(0))
    return Q(k0)+gamma*phi,k0,phi


def lp_bound(edges,nodes,entries,traces,rho,eta,p,gamma,k0,phi):
    import numpy as np
    from scipy.optimize import linprog
    edges=sorted(edges);m=len(edges);n=len(traces)
    caps=Counter(u for u,v in edges)
    hinge=[(u,j) for u in sorted(caps) for j in range(1,caps[u])]
    dim=m+n+len(hinge);obj=np.zeros(dim);obj[:m]=1
    for h,(u,j) in enumerate(hinge):obj[m+n+h]=float(gamma*p.get(u,Q(0)))
    mat=[];rhs=[]
    for i,row in enumerate(traces):
        a=np.zeros(dim);a[m+i]=1
        for j,e in enumerate(edges):a[j]=-float(Q(row[e],1)/(rho*sum(row.values())))
        mat.append(a);rhs.append(0)
    a=np.zeros(dim);a[m:m+n]=-1;mat.append(a);rhs.append(-float(n*eta))
    for h,(u,j) in enumerate(hinge):
        a=np.zeros(dim)
        for z,(s,t) in enumerate(edges):
            if s==u:a[z]=1
        a[m+n+h]=-1;mat.append(a);rhs.append(j)
    nonentries=sorted(set(nodes)-set(entries))
    for k in range(1,len(nonentries)+1):
        for subset in combinations(nonentries,k):
            ss=set(subset);a=np.zeros(dim)
            for j,(u,v) in enumerate(edges):
                if v in ss and u not in ss:a[j]=-1
            mat.append(a);rhs.append(-1)
    # Both valid global cuts make the LP dominate the elementary bound.
    a=np.zeros(dim);a[:m]=-1;mat.append(a);rhs.append(-k0)
    a=np.zeros(dim)
    for h,(u,j) in enumerate(hinge):a[m+n+h]=-float(p.get(u,Q(0)))
    mat.append(a);rhs.append(-float(phi))
    res=linprog(obj,A_ub=np.asarray(mat),b_ub=np.asarray(rhs),
                bounds=[(0,1)]*(m+n)+[(0,None)]*len(hinge),method='highs')
    assert res.success,res.message
    return res.fun


def check_random(seed=20261021,instances=32):
    rng=random.Random(seed);answer=[];inspected=0
    while len(answer)<instances:
        nv=rng.choice([3,4,5]);nodes=set(range(nv))
        walks=[[rng.randrange(nv) for _ in range(rng.randrange(3,7))] for _ in range(4)]
        seen=set(v for walk in walks for v in walk)
        if seen!=nodes:continue
        traces=[Counter(zip(walk,walk[1:])) for walk in walks]
        edges=set().union(*(set(row) for row in traces))
        if len(edges)>10:continue
        entries={walk[0] for walk in walks}
        rho=rng.choice([Q(1,2),Q(4,5),Q(1)])
        eta=rng.choice([Q(3,5),Q(17,20),Q(1)])
        gamma=rng.choice([Q(0),Q(1,2),Q(2)])
        counts=Counter()
        for row in traces:
            for (u,v),n in row.items():counts[u]+=n
        total=sum(counts.values());p={u:Q(n,total) for u,n in counts.items()}
        bound,k0,phi=lower_bound(edges,nodes,entries,traces,rho,eta,p,gamma)
        all_feasible=[];es=sorted(edges)
        for mask in range(1<<len(es)):
            selected={e for j,e in enumerate(es) if mask>>j&1};inspected+=1
            if coverage(selected,traces,rho)>=eta and reachable(selected,nodes,entries):
                val=cost(selected,p,gamma);assert bound<=val
                all_feasible.append((val,selected))
        optimum=min(v for v,s in all_feasible)
        lp=lp_bound(edges,nodes,entries,traces,rho,eta,p,gamma,k0,phi)
        assert float(bound)-1e-7<=lp<=float(optimum)+1e-7,(bound,lp,optimum)
        # Check diminishing coverage gains and increasing cost gains exactly.
        for _ in range(12):
            ee=rng.choice(es)
            bigger={e for e in es if e!=ee and rng.random()<.7}
            smaller={e for e in bigger if rng.random()<.5}
            assert coverage(smaller|{ee},traces,rho)-coverage(smaller,traces,rho)>=coverage(bigger|{ee},traces,rho)-coverage(bigger,traces,rho)
            assert cost(smaller|{ee},p,gamma)-cost(smaller,p,gamma)<=cost(bigger|{ee},p,gamma)-cost(bigger,p,gamma)
        # Follow feasible, strictly descending deletion/exchange steps.
        selected=set(edges);values=[cost(selected,p,gamma)]
        for _ in range(8):
            best=None
            for removed in sorted(selected):
                for added in [None]+sorted(edges-selected):
                    nxt=(selected-{removed})|({added} if added is not None else set())
                    if coverage(nxt,traces,rho)<eta or not reachable(nxt,nodes,entries):continue
                    val=cost(nxt,p,gamma)
                    if val<values[-1] and (best is None or val<best[0]):best=(val,nxt)
            if best is None:break
            val,selected=best;values.append(val)
        assert all(values[i+1]<values[i] for i in range(len(values)-1))
        assert all(bound<=optimum<=val for val in values)
        assert all(bound/values[i+1]>=bound/values[i] for i in range(len(values)-1))
        answer.append({'edges':len(edges),'feasible_subsets':len(all_feasible),'lower_bound':float(bound),
                       'lp_lower_bound':lp,'exact_optimum':float(optimum),
                       'accepted_moves':len(values)-1,'final_cost':float(values[-1])})
    return answer,inspected


def check_counterexamples():
    result=[]
    for n in [2,4,8,16]:
        common=('s','t');connectors=[('t',f'u{i}') for i in range(n)]
        tails=[(f'u{i}',f'v{i}') for i in range(n)]
        edges={common,*connectors,*tails};nodes={v for e in edges for v in e}
        # Single-action observations make every node a legitimate entry; such
        # observations contribute no transition and are excluded from D+.
        entries=set(nodes);traces=[Counter({common:1,connectors[i]:1,tails[i]:1}) for i in range(n)]
        rho,eta,gamma=Q(1,3),Q(1),Q(0);p={}
        selected=set(edges)
        for removed in [common]+connectors:
            nxt=selected-{removed}
            assert coverage(nxt,traces,rho)==1 and reachable(nxt,nodes,entries)
            assert cost(nxt,p,gamma)<cost(selected,p,gamma);selected=nxt
        assert selected==set(tails)
        assert all(coverage(selected-{e},traces,rho)<eta for e in selected)
        # Every one-for-one exchange has the same cardinality and hence cost.
        for old in selected:
            for new in edges-selected:
                assert cost((selected-{old})|{new},p,gamma)==n
        assert coverage({common},traces,rho)==1
        bound,_,_=lower_bound(edges,nodes,entries,traces,rho,eta,p,gamma)
        assert bound==1
        result.append({'n':n,'local_cost':n,'exact_optimum':1,'ratio':n})
    return result


if __name__=='__main__':
    rows,inspected=check_random();bad=check_counterexamples()
    output=Path(__file__).resolve().parents[1]/'refine-logs'/'treefree-tracecover'/'LOWER_BOUND_SANITY_20261021.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'scope':'synthetic exact arithmetic and numerical LP checks; no production miner changes',
            'seed':20261021,'instances':len(rows),'subsets_inspected':inspected,
            'instance_results':rows,'strict_local_search_counterexamples':bad,
            'lp_note':'Floating-point LP sanity only; not a rigorous numerical dual certificate.'}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'instances':len(rows),'subsets_inspected':inspected,
                      'all_checks_passed':True,'counterexample_ratios':[r['ratio'] for r in bad],
                      'report':str(output)},indent=2))
