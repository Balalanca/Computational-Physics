import numpy as np

def logistic(x,mu):return 4*mu*x*(1-x)
def trajectory(g,x,mu,n,burn=0):
    for _ in range(burn):x=g(x,mu)
    out=np.empty(n)
    for i in range(n):out[i]=x;x=g(x,mu)
    return out

def lyapunov(mu,n=10000,x=.1234567):
    t=trajectory(logistic,x,mu,n,1000);der=np.abs(4*mu*(1-2*t));return np.mean(np.log(np.maximum(der,np.finfo(float).tiny)))
def dimensions(t,eps):
    counts=[];entropy=[]
    for e in eps:
        p=np.histogram(t,bins=np.arange(0,1+e,e))[0];p=p[p>0]/len(t);counts.append(len(p));entropy.append(-np.sum(p*np.log(p)))
    x=np.log(1/np.asarray(eps));return np.polyfit(x,np.log(counts),1)[0],np.polyfit(x,entropy,1)[0]


def separation(mu, x0=0.1, delta=1e-13, n=100):
    """Distance between nearby trajectories before nonlinear saturation."""
    return np.abs(trajectory(logistic,x0,mu,n)-trajectory(logistic,x0+delta,mu,n))


def fit_separation(distance, lower=1e-12, upper=1e-2):
    distance=np.asarray(distance)
    valid=(distance>lower)&(distance<upper)
    if valid.sum()<3:
        raise ValueError('insufficient separation data')
    return np.polyfit(np.flatnonzero(valid),np.log(distance[valid]),1)


def dimension_curve(t, eps):
    """Local box-counting and Shannon entropy slopes."""
    eps=np.asarray(eps,float)
    counts, entropy=[],[]
    for e in eps:
        hist=np.histogram(t,bins=np.r_[np.arange(0,1,e),1.])[0]
        p=hist[hist>0]/hist.sum()
        counts.append(len(p)); entropy.append(-np.sum(p*np.log(p)))
    scale=np.log(1/eps)
    return np.sqrt(eps[:-1]*eps[1:]),np.diff(np.log(counts))/np.diff(scale),np.diff(entropy)/np.diff(scale)


def critical_orbit(mu,n=20):
    return trajectory(logistic,.5,mu,n+1)[1:]


def invariant_density(mu,n=1000000,bins=200):
    t=trajectory(logistic,.1234567,mu,n,10000)
    h,e=np.histogram(t,bins=bins,range=(0,1),density=True)
    return (e[:-1]+e[1:])/2,h
