import numpy as np
from ising import energy,clusters,sweep
from percolation import site,largest_fraction,bond,box_dimension,urban_scan
from avalanches import simulate
from chaos import lyapunov,trajectory
from hopfield import Hopfield,modern_retrieve

def test_ising():
    s=np.ones((4,4),int);assert energy(s)==-32;assert clusters(s).max()==1;assert np.isin(sweep(s,2,np.random.default_rng(1)),[-1,1]).all()
def test_percolation():
    rng=np.random.default_rng(1);assert largest_fraction(site(8,0,rng))==0;assert largest_fraction(site(8,1,rng))==1;assert bond(4,1,rng)==[16];assert np.isclose(box_dimension(np.ones((32,32),bool)),2)
def test_avalanche():
    for seed in range(5):
        h=np.random.default_rng(seed).normal(size=(9,10));f,s=simulate(h,'brute');g,t=simulate(h,'sorted');assert np.allclose(f,g);assert np.array_equal(s,t);assert s.sum()==90;assert np.all(np.diff(f)>=-1e-10)
def test_chaos():
    assert abs(lyapunov(1)-np.log(2))<.02;assert np.allclose(trajectory(lambda x,m:x+m,0,2,3),[0,2,4])
def test_hopfield():
    p=np.array([[1,-1,1,-1,1,-1]]);x=p[0].copy();x[0]*=-1;y,h=Hopfield(p).retrieve(x,np.random.default_rng(1));assert np.array_equal(y,p[0]);assert np.all(np.diff(h)<=1e-12);assert np.allclose(modern_retrieve(p,x),p[0])
