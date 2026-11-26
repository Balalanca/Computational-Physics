import numpy as np
from collections import deque
from time import perf_counter

def simulate(h,method='sorted',J=1.):
    if method not in ('sorted','brute') or J<0:raise ValueError('invalid method/coupling')
    h=np.asarray(h,float);n,m=h.shape;s=-np.ones(h.shape,int);up=np.zeros(h.shape,int);events=[];fields=[];remaining=s.size
    def neighbors(i,j):return [((i-1)%n,j),((i+1)%n,j),(i,(j-1)%m),(i,(j+1)%m)]
    order=np.argsort(h.ravel())[::-1] if method=='sorted' else None;ptr=np.zeros(5,int)
    while remaining:
        if method=='brute':
            local=np.where(s<0,J*(2*up-4)+h,-np.inf);i,j=np.unravel_index(np.argmax(local),h.shape);H=-local[i,j]
        else:
            while True:
                candidates=[]
                for k in range(5):
                    while ptr[k]<s.size and s.flat[order[ptr[k]]]>0:ptr[k]+=1
                    if ptr[k]<s.size:
                        idx=order[ptr[k]];candidates.append((J*(2*k-4)+h.flat[idx],k,idx))
                val,k,idx=max(candidates);ptr[k]+=1;i,j=np.unravel_index(idx,h.shape)
                if up[i,j]==k: H=-val;break
        q=deque([(i,j)]);count=0
        while q:
            a,b=q.popleft()
            if s[a,b]>0:continue
            s[a,b]=1;count+=1;remaining-=1
            for x,y in neighbors(a,b):
                up[x,y]+=1
                if s[x,y]<0 and J*(2*up[x,y]-4)+h[x,y]+H>=-1e-12:q.append((x,y))
        events.append(count);fields.append(H)
    return np.array(fields),np.array(events)

def benchmark(h):
    result={}
    for method in ('brute','sorted'):
        start=perf_counter();f,s=simulate(h,method);result[method]=(perf_counter()-start,f,s)
    if not np.allclose(result['brute'][1],result['sorted'][1]) or not np.array_equal(result['brute'][2],result['sorted'][2]):raise AssertionError('algorithms disagree')
    return result
