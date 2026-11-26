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
