from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ising import sweep,schelling
from percolation import site,largest_fraction,bond,box_dimension,urban_scan,sizes
from avalanches import benchmark
from chaos import logistic,trajectory,lyapunov
from hopfield import Hopfield,modern_retrieve

def main():
    p=argparse.ArgumentParser();p.add_argument('--urban-grid');a=p.parse_args();out=Path('results');out.mkdir(exist_ok=True);rng=np.random.default_rng(42)
    fig,ax=plt.subplots();temps=[1.5,2.,2.27,2.6,3.5]
    for n in [16,24]:
        ms=[]
        for T in temps:
            s=rng.choice([-1,1],size=(n,n));samples=[]
            for j in range(180):
                sweep(s,T,rng)
                if j>=100 and j%4==0:samples.append(abs(s.mean()))
            ms.append(np.mean(samples))
        ax.plot(temps,ms,'o-',label=f'L={n}')
    ax.set(xlabel='Temperature',ylabel='Mean absolute magnetization');ax.legend();fig.savefig(out/'ising.png');plt.close(fig)
    fig,ax=plt.subplots();ps=np.linspace(.45,.75,15)
    for n in [16,32,64]:ax.plot(ps,[np.mean([largest_fraction(site(n,p,rng)) for _ in range(20)]) for p in ps],label=f'L={n}')
    ax.legend();ax.set(xlabel='Occupation probability',ylabel='Largest cluster fraction');fig.savefig(out/'percolation.png');plt.close(fig)
    lab=site(512,.59274,rng);print('Fractal dimension',box_dimension(lab==np.argmax(sizes(lab))+1));print('Bond largest component',bond(40,.5,rng)[0]);plt.imsave(out/'schelling.png',schelling(50,.6,rng),cmap='coolwarm')
    fig,ax=plt.subplots()
    for R in [0.7,0.9,1.4]:
        b=benchmark(rng.normal(0,R,(60,60)));print('Matched benchmark',R,{k:v[0] for k,v in b.items()});_,f,s=b['sorted'];ax.step(f,2*np.cumsum(s)/3600-1,label=f'R={R}')
    ax.legend();ax.set(xlabel='Field',ylabel='Magnetization per spin');fig.savefig(out/'avalanches.png');plt.close(fig)
    mus=np.linspace(.8,1,400);fig,ax=plt.subplots();ax.plot(np.repeat(mus,100),np.concatenate([trajectory(logistic,.1234,m,100,500) for m in mus]),',k');ax.set(xlabel='mu',ylabel='x');fig.savefig(out/'chaos.png');plt.close(fig);print('Lyapunov at mu=1',lyapunov(1.))
    patterns=rng.choice([-1,1],size=(4,100));net=Hopfield(patterns);x=patterns[0].copy();x[rng.choice(100,20,replace=False)]*=-1;y,h=net.retrieve(x,rng);print('Classical retrieval overlap',y@patterns[0]/100,'modern overlap',modern_retrieve(patterns,x)@patterns[0]/100);fig,ax=plt.subplots();ax.plot(h);ax.set(xlabel='Neuron updates',ylabel='Energy');fig.savefig(out/'hopfield.png');plt.close(fig)
    if a.urban_grid:
        grid=np.loadtxt(a.urban_grid);t=np.linspace(0,5000,101);f,d=urban_scan(grid,t);np.savetxt(out/'urban_scan.csv',np.column_stack((t,f)),delimiter=',',header='threshold,largest_land_fraction',comments='');print('Steepest interval',t[d.argmax():d.argmax()+2])
if __name__=='__main__':main()
