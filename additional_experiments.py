"""Additional finite-size, spatial-cluster, and image-memory investigations."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,label
from ising import sweep,energy
from percolation import site,bond_labels,cluster_density,sizes,order_curve
from hopfield import MemoryDynamics,modern_history
from restored_experiments import save,table


def magnetization(out,rng):
    # Start ordered to reduce low-temperature domain-trapping; compare to exact infinite-volume result.
    temperatures=np.linspace(1.5,3.5,12);tc=2/np.log(1+np.sqrt(2));fig,ax=plt.subplots();rows=[]
    for n in (16,32):
        for T in temperatures:
            chain=[];e=[]
            for run in range(3):
                s=np.ones((n,n),dtype=int)
                for j in range(80):
                    sweep(s,T,rng)
                    if j>=40 and j%4==0:chain.append(abs(s.mean()));e.append(energy(s)/s.size)
            rows.append([n,T,np.mean(chain),np.mean(e)])
        a=np.array(rows);a=a[a[:,0]==n];ax.plot(a[:,1],a[:,2],'o-',label=f'L={n}')
    t=np.linspace(1.5,tc-.00001,200);ax.plot(t,(1-np.sinh(2/t)**-4)**.125,'--',label='Exact infinite lattice');ax.axvline(tc,ls=':',label='Critical T');ax.legend();ax.set(xlabel='Temperature (J/kB)',ylabel='Mean absolute magnetization');save(fig,out,'magnetization');table(out,'magnetization',rows,'size,temperature,magnetization,energy_per_spin')


def percolation_details(out,rng):
    fig,axes=plt.subplots(2,6,figsize=(15,5))
    for row,(model,pc) in enumerate(((site,.59274),(bond_labels,.5))):
        for ax,p in zip(axes[row],[.2,.4,pc,.65,.8,.95]):
            a=model(60,p,rng);ax.imshow(a%31,cmap='nipy_spectral');ax.set_title(f'p={p:.3f}');ax.axis('off')
    save(fig,out,'cluster_configurations')
    fig,ax=plt.subplots()
    for model,pc,name in [(site,.59274,'Site'),(bond_labels,.5,'Bond')]:
        dp=2.**-np.arange(2,9);a=order_curve(model,120,pc+dp,rng,10);ax.loglog(dp,a[:,1],'o-',label=name);table(out,f'{name.lower()}_supercritical',a,'p,largest_fraction,se')
    ax.loglog(dp,.5*(dp/dp[3])**(5/36),'--',label='5/36 reference');ax.set(xlabel='p - pc',ylabel='Largest cluster fraction');ax.legend();save(fig,out,'supercritical_order')
    # Reproducible correlated artificial population field, explicitly not geographic data.
    y,x=np.mgrid[:200,:300];grid=100*np.exp(gaussian_filter(rng.normal(size=x.shape),3)*8)
    for a,b in rng.uniform([0,0],[300,200],(30,2)):grid+=rng.uniform(500,3000)*np.exp(-((x-a)**2+(y-b)**2)/rng.uniform(30,300))
    grid[:,:20]=-1;fig,ax=plt.subplots()
    for t in (10,50,200,500,1000,2000):
        z,_=label((grid>=0)&(grid>t));xx,yy=cluster_density(sizes(z));ax.loglog(xx,yy,'o-',label=str(t))
    ax.set(title='Synthetic population field',xlabel='Cluster area',ylabel='Cluster density');ax.legend(title='Threshold');save(fig,out,'urban_synthetic_clusters')


def image_memory(out,rng):
    from sklearn.datasets import load_digits
    # The bundled 8x8 digits offer a redistributable benchmark without private image inputs.
    data=load_digits();patterns=np.array([np.where(data.images[np.flatnonzero(data.target==i)[0]]>7,1,-1).ravel() for i in range(6)])
    net=MemoryDynamics(patterns);fig,axes=plt.subplots(3,6,figsize=(10,5))
    for i,p in enumerate(patterns):
        x=p.copy();x[rng.choice(len(x),16,replace=False)]*=-1
        states,_,_=net.evolve(x,rng,1000);modern,_=modern_history(patterns,x,.2,30)
        for ax,z in zip(axes[:,i],[x,states[-1],modern[-1]]):ax.imshow(z.reshape(8,8),cmap='gray',vmin=-1,vmax=1);ax.axis('off')
    fig.suptitle('Corrupted input / classical recall / modern recall');save(fig,out,'digit_memory')
    rows=[]
    for count in [2,4,6]:
        p=patterns[:count];model=MemoryDynamics(p)
        for noise in (0,.1,.2,.3,.4,.5):
            for trial in range(10):
                idx=int(rng.integers(count));x=p[idx].copy();x[rng.choice(64,int(64*noise),replace=False)]*=-1
                s,_,_=model.evolve(x,rng,600);m,_=modern_history(p,x,.2,30)
                rows.append([count,noise,np.mean(s[-1]==p[idx]),np.mean(np.sign(m[-1])==p[idx])])
    a=np.array(rows);table(out,'digit_retrieval',a,'patterns,noise,classical_pixel_accuracy,modern_pixel_accuracy')
    fig,ax=plt.subplots()
    for count in [2,4,6]:
        b=a[a[:,0]==count];noise=np.unique(b[:,1]);ax.plot(noise,[b[b[:,1]==x,2].mean() for x in noise],'o-',label=f'Classical {count}');ax.plot(noise,[b[b[:,1]==x,3].mean() for x in noise],'s--',label=f'Modern {count}')
    ax.set(xlabel='Corrupted fraction',ylabel='Mean pixel accuracy');ax.legend(fontsize=8);save(fig,out,'digit_noise_sweep')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['all','magnetization','percolation','memory'],default='all');a=p.parse_args();out=Path('results/extended');out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('magnetization',magnetization),('percolation',percolation_details),('memory',image_memory)]:
        if a.study in ('all',name):print(name,flush=True);fn(out,np.random.default_rng(42))
