import numpy as np
from ising import energy,metropolis_update,quench,schelling_history,cluster_sizes
from percolation import spanning,bond_labels,cluster_density,urban_statistics
from avalanches import simulate
from chaos import dimension_curve,invariant_density,critical_orbit,logistic
from hopfield import MemoryDynamics,modern_history

def test_energy_updates():
    rng=np.random.default_rng(7);s=rng.choice([-1,1],(8,9));e=energy(s)
    for _ in range(500):
        e+=metropolis_update(s,2.3,rng)
        assert e==energy(s)

def test_quench_measurements():
    rows,states=quench(8,[0,10,100],np.random.default_rng(1))
    assert len(states)==3 and np.all(rows[:,3]>=0)
    for row,state in zip(rows,states):
        assert row[3]==(energy(state)+2*state.size)/2

def test_population_conservation():
    rows,states=schelling_history(9,.6,[0,10,1000],np.random.default_rng(3))
    for s in states:
        assert np.array_equal(np.bincount(s.ravel()+1),np.bincount(states[0].ravel()+1))
        assert cluster_sizes(s,True).sum()==np.count_nonzero(s)

def test_spanning_and_bonds():
    rng=np.random.default_rng(7)
    assert spanning(np.eye(4,dtype=int)*np.arange(1,5))==(False,False)
    assert spanning(bond_labels(4,1,rng))==(True,True)
    assert len(np.unique(bond_labels(4,0,rng)))==16

def test_urban_threshold():
    a=urban_statistics(np.array([[0,2],[1,-1.]]),[0,1,2])
    assert np.all(np.diff(a[:,1])<=0) and a[-1,3]==0

def test_avalanche_maps():
    h=np.random.default_rng(8).normal(size=(12,13));f,s,m=simulate(h,return_map=True)
    assert np.array_equal(np.bincount(m.ravel())[1:],s)
    assert s.sum()==h.size

def test_dimension_limits():
    eps=2.**np.arange(-3,-8,-1);_,d,i=dimension_curve(np.full(100,.5),eps)
    assert np.allclose(d,0) and np.allclose(i,0)
    _,d,i=dimension_curve(np.linspace(0,1,4096,endpoint=False),eps)
    assert np.allclose(d,1) and np.allclose(i,1)

def test_invariant_density():
    x,h=invariant_density(1.,100000,100)
    assert np.isclose(h.sum()/100,1) and abs(np.sum(h*x)/100-.5)<.01
    assert np.isclose(critical_orbit(.9,2)[-1],logistic(.9,.9))

def test_memory_dynamics():
    p=np.random.default_rng(3).choice([-1,1],(3,20));net=MemoryDynamics(p)
    s,e,o=net.evolve(-p[0],np.random.default_rng(4));assert np.all(np.diff(e)<=1e-12);assert np.allclose(o[0,0],-1)
    _,e=modern_history(p,p[0]*.4,.3);assert np.all(np.diff(e)<=1e-10)
