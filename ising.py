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


def metropolis_update(s,T,rng):
    """One random-site update; return actual energy change."""
    if T<=0: raise ValueError('positive temperature required')
    n,m=s.shape
    i,j=rng.integers(n),rng.integers(m)
    delta=2*int(s[i,j])*int(s[(i-1)%n,j]+s[(i+1)%n,j]+s[i,(j-1)%m]+s[i,(j+1)%m])
    if delta<=0 or rng.random()<np.exp(-delta/T):
        s[i,j]*=-1
        return delta
    return 0


def energy_history(n,T,updates,rng):
    s=rng.choice([-1,1],size=(n,n)); e=energy(s); values=[e]
    for _ in range(updates):
        e+=metropolis_update(s,T,rng);values.append(e)
    return np.asarray(values)


def cluster_sizes(s, exclude_empty=False):
    labels=clusters(s)
    return np.bincount(labels[s!=0] if exclude_empty else labels.ravel())[1:][np.bincount(labels[s!=0] if exclude_empty else labels.ravel())[1:]>0]


def quench(n,times,rng,high=10.,low=1.):
    s=rng.choice([-1,1],size=(n,n))
    for _ in range(10): sweep(s,high,rng)
    states=[];rows=[];t=0
    for target in times:
        while t<target: metropolis_update(s,low,rng);t+=1
        a=cluster_sizes(s); walls=(energy(s)+2*s.size)/2
        rows.append([target,a.mean(),np.sqrt(a.mean()),walls]); states.append(s.copy())
    return np.array(rows),states


def schelling_history(n,threshold,times,rng,empty_fraction=.2):
    """Track occupied clusters and relocation counts at specified update times."""
    vacancies=int(n*n*empty_fraction);a=(n*n-vacancies)//2
    s=np.array([1]*a+[-1]*(n*n-vacancies-a)+[0]*vacancies).reshape(n,n)
    rng.shuffle(s.ravel());empty=list(map(tuple,np.argwhere(s==0)))
    states=[];rows=[];t=moves=0
    for target in times:
        while t<target:
            i,j=rng.integers(n,size=2);v=s[i,j];t+=1
            if v==0 or not empty:continue
            ns=np.array([s[(i+a)%n,(j+b)%n] for a in [-1,0,1] for b in [-1,0,1] if (a,b)!=(0,0)])
            occupied=ns[ns!=0]
            if len(occupied) and np.mean(occupied==v)<threshold:
                k=rng.integers(len(empty));a,b=empty[k];s[a,b]=v;s[i,j]=0;empty[k]=(i,j);moves+=1
        rows.append([target,cluster_sizes(s,True).mean(),moves]);states.append(s.copy())
    return np.array(rows),states
