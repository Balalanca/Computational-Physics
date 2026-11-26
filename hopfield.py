import numpy as np
from scipy.special import softmax

class Hopfield:
    def __init__(self,patterns):
        self.patterns=np.asarray(patterns,float)
        if self.patterns.ndim!=2 or not np.all(np.isin(self.patterns,[-1,1])):raise ValueError('expected bipolar patterns')
        self.w=self.patterns.T@self.patterns/self.patterns.shape[1];np.fill_diagonal(self.w,0)
    def energy(self,x):return -.5*x@self.w@x
    def retrieve(self,x,rng,sweeps=20):
        x=np.asarray(x,float).copy();history=[self.energy(x)]
        for _ in range(sweeps):
            for i in rng.permutation(len(x)):
                h=self.w[i]@x;x[i]=1 if h>0 else -1 if h<0 else x[i];history.append(self.energy(x))
        return x,np.array(history)

def modern_retrieve(patterns,x,beta=2.,steps=20):
    p=np.asarray(patterns,float);x=np.asarray(x,float).copy()
    for _ in range(steps):x=softmax(beta*(p@x))@p
    return x


class MemoryDynamics(Hopfield):
    """Classical memory with explicit synchronous/asynchronous dynamics and history."""
    def evolve(self,state,rng,steps=100,synchronous=False):
        x=np.asarray(state,float).copy();states=[x.copy()]
        for _ in range(steps):
            if synchronous:
                h=self.w@x;x=np.where(h==0,x,np.sign(h))
            else:
                i=rng.integers(len(x));h=self.w[i]@x
                if h!=0:x[i]=np.sign(h)
            states.append(x.copy())
        states=np.array(states)
        norms=np.linalg.norm(states,axis=1)[:,None]*np.linalg.norm(self.patterns,axis=1)[None,:]
        overlap=np.divide(states@self.patterns.T,norms,out=np.zeros_like(norms),where=norms>0)
        return states,np.array([self.energy(x) for x in states]),overlap


def modern_history(patterns,state,beta,steps=20):
    from scipy.special import logsumexp
    if beta<=0:raise ValueError('beta must be positive')
    p=np.asarray(patterns,float);x=np.asarray(state,float).copy();states=[x.copy()]
    for _ in range(steps):
        x=softmax(beta*(p@x))@p;states.append(x.copy())
    return np.array(states),np.array([.5*x@x-logsumexp(beta*(p@x))/beta for x in states])
