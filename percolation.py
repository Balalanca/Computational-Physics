import numpy as np
from scipy.ndimage import label
import networkx as nx

def site(n,p,rng):return label(rng.random((n,n))<p)[0]
def sizes(labels):return np.bincount(labels.ravel())[1:]
def largest_fraction(labels):
    a=sizes(labels);return a.max()/labels.size if len(a) else 0.
def bond(n,p,rng):
    g=nx.grid_2d_graph(n,n);g.remove_edges_from([e for e in g.edges if rng.random()>=p]);return sorted((len(c) for c in nx.connected_components(g)),reverse=True)
def box_dimension(mask):
    n=mask.shape[0]
    if mask.shape!=(n,n) or n&(n-1) or not mask.any():raise ValueError('nonempty power-of-two square required')
    b=2**np.arange(1,int(np.log2(n)));c=np.array([mask.reshape(n//x,x,n//x,x).any(axis=(1,3)).sum() for x in b]);return -np.polyfit(np.log(b),np.log(c),1)[0]
def urban_scan(grid,thresholds):
    land=grid>=0
    if not land.any():raise ValueError('no land')
    t=np.asarray(thresholds,float)
    if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('thresholds must increase')
    fractions=[]
    for x in t:
        a=sizes(label(land&(grid>x))[0]);fractions.append(a.max()/land.sum() if len(a) else 0.)
    f=np.array(fractions);return f,-np.diff(f)/np.diff(t)


def bond_labels(n,p,rng):
    """Label connected components, including isolated sites."""
    g=nx.grid_2d_graph(n,n)
    g.remove_edges_from([e for e in g.edges if rng.random()>=p])
    out=np.zeros((n,n),int)
    for k,c in enumerate(nx.connected_components(g),1):
        for i,j in c:out[i,j]=k
    return out


def cluster_density(a,bins=30):
    a=np.asarray(a);a=a[a>0]
    if not len(a):return np.array([]),np.array([])
    edges=np.geomspace(1,max(2,a.max()+1),bins+1)
    h,e=np.histogram(a,edges);valid=h>0
    return np.sqrt(e[:-1]*e[1:])[valid],(h/(len(a)*np.diff(e)))[valid]


def order_curve(model,n,probabilities,rng,repeats=15):
    out=[]
    for p in probabilities:
        x=np.array([largest_fraction(model(n,p,rng)) for _ in range(repeats)])
        out.append([p,x.mean(),x.std(ddof=1)/np.sqrt(repeats)])
    return np.array(out)


def spanning(labels):
    def common(a,b):return bool((set(a)&set(b))-{0})
    return common(labels[0],labels[-1]),common(labels[:,0],labels[:,-1])


def urban_statistics(grid,thresholds):
    land=np.isfinite(grid)&(grid>=0)
    if not land.any():raise ValueError('no land')
    rows=[]
    for threshold in thresholds:
        mask=land&(grid>threshold);labels,_=label(mask);a=sizes(labels);v,h=spanning(labels)
        rows.append([threshold,mask.sum()/land.sum(),len(a),a.max()/land.sum() if len(a) else 0,a.mean() if len(a) else 0,v,h])
    return np.array(rows)
