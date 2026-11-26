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
