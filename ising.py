import numpy as np

def energy(s): return -int(np.sum(s*(np.roll(s,1,0)+np.roll(s,1,1))))
def sweep(s,T,rng):
    if T<=0: raise ValueError('T must be positive')
    n,m=s.shape
    for _ in range(s.size):
        i,j=rng.integers(n),rng.integers(m);de=2*s[i,j]*(s[(i-1)%n,j]+s[(i+1)%n,j]+s[i,(j-1)%m]+s[i,(j+1)%m])
        if de<=0 or rng.random()<np.exp(-de/T):s[i,j]*=-1
    return s

def clusters(s):
    n,m=s.shape;labels=np.zeros(s.shape,int);k=0
    for i,j in np.ndindex(s.shape):
        if labels[i,j]:continue
        k+=1;labels[i,j]=k;stack=[(i,j)]
        while stack:
            a,b=stack.pop()
            for x,y in [((a-1)%n,b),((a+1)%n,b),(a,(b-1)%m),(a,(b+1)%m)]:
                if not labels[x,y] and s[x,y]==s[i,j]:labels[x,y]=k;stack.append((x,y))
    return labels

def schelling(n,threshold,rng,steps=10000):
    if n<2 or not 0<=threshold<=1:raise ValueError('invalid parameters')
    s=rng.choice([-1,0,1],size=(n,n),p=[.4,.2,.4]);empty=list(map(tuple,np.argwhere(s==0)))
    for _ in range(steps):
        i,j=rng.integers(n,size=2);v=s[i,j]
        if v==0 or not empty:continue
        ns=[s[(i+a)%n,(j+b)%n] for a in [-1,0,1] for b in [-1,0,1] if (a,b)!=(0,0)];occupied=[x for x in ns if x]
        if occupied and np.mean(np.array(occupied)==v)<threshold:
            k=rng.integers(len(empty));a,b=empty[k];s[a,b]=v;s[i,j]=0;empty[k]=(i,j)
    return s
