"""Extended experiments for statistical mechanics and nonlinear dynamics."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ising import energy_history, quench, schelling_history, sweep, cluster_sizes
from percolation import site, bond_labels, order_curve, cluster_density, sizes, urban_statistics, box_dimension
from avalanches import simulate, benchmark
from chaos import trajectory, logistic, separation, fit_separation, dimension_curve, invariant_density, critical_orbit
from hopfield import MemoryDynamics, modern_history

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=140);plt.close(fig)

def table(out,name,data,header):
    np.savetxt(out/(name+'.csv'),data,delimiter=',',header=header,comments='')

def ising_study(out,rng,quick):
    fig,ax=plt.subplots()
    for T in (1.,2.269185,10.):
        e=energy_history(20,T,5000 if quick else 50000,rng)
        ax.semilogx(np.arange(1,len(e)),e[1:]/400,label=f'T={T:g}')
    ax.set(xlabel='Updates',ylabel='Energy per spin');ax.legend();save(fig,out,'energy_evolution')
    fig,ax=plt.subplots()
    for T in (2.269185,10.):
        n=24 if quick else 40;s=rng.choice([-1,1],(n,n));data=[]
        for i in range(30 if quick else 200):
            sweep(s,T,rng)
            if i>=10:data.extend(cluster_sizes(s))
        x,y=cluster_density(data);ax.loglog(x,y,'o-',label=f'T={T:g}')
    ax.set(xlabel='Cluster area',ylabel='Density');ax.legend();save(fig,out,'ising_clusters')
    n=30 if quick else 100;times=np.r_[0,np.unique(np.geomspace(1,(5 if quick else 50)*n*n,18).astype(int))]
    rows,states=quench(n,times,rng);table(out,'quench',rows,'updates,mean_area,sqrt_mean_area,domain_wall_bonds')
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,i in zip(axes,np.linspace(0,len(times)-1,4).astype(int)):
        ax.imshow(states[i],cmap='gray');ax.set_title(f'{times[i]} updates');ax.axis('off')
    save(fig,out,'quench_snapshots')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k,label in zip(axes,[2,3],['Sqrt(mean cluster area)','Domain wall bonds']):
        ax.loglog(rows[1:,0]/n**2,rows[1:,k]);ax.set(xlabel='Sweeps',ylabel=label)
    save(fig,out,'coarsening')
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for col,B in enumerate((.3,.5,.7)):
        rows,states=schelling_history(40,B,[0,100,1000,10000,40000],rng)
        axes[0,col].imshow(states[-1],cmap='coolwarm',vmin=-1,vmax=1);axes[0,col].axis('off');axes[0,col].set_title(f'B={B}')
        axes[1,col].semilogx(rows[1:,0],rows[1:,1]);axes[1,col].set(xlabel='Updates',ylabel='Occupied cluster area');table(out,f'schelling_{B}',rows,'updates,mean_cluster_area,moves')
    save(fig,out,'segregation')

def percolation_study(out,rng,quick,urban_grid):
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for row,(model,pc,name) in enumerate(((site,.59274,'site'),(bond_labels,.5,'bond'))):
        for n in (5,10,20,50,100):
            p=np.clip(pc+np.linspace(-2,2,11)*n**(-.75),0,1);a=order_curve(model,n,p,rng,5 if quick else 20)
            axes[row,0].errorbar(a[:,0],a[:,1],yerr=a[:,2],label=str(n));axes[row,1].plot((p-pc)*n**.75,a[:,1]*n**(5/48),'o-',label=str(n));table(out,f'{name}_order_{n}',a,'p,largest_fraction,se')
        labels=model(100 if quick else 400,pc,rng);x,y=cluster_density(sizes(labels));axes[row,2].loglog(x,y,'o',label=name);axes[row,2].loglog(x,y[0]*(x/x[0])**(-187/91),'--',label='187/91 reference')
        for ax in axes[row]:ax.legend(fontsize=7)
        axes[row,0].set(xlabel='p',ylabel=name+' order parameter');axes[row,1].set(xlabel='Scaled p',ylabel='Scaled order parameter');axes[row,2].set(xlabel='Cluster area',ylabel='Density')
    save(fig,out,'finite_size_scaling')
    lab=site(512,.59274,rng);mask=lab==np.argmax(sizes(lab))+1;b=2**np.arange(1,9);c=np.array([mask.reshape(512//i,i,512//i,i).any(axis=(1,3)).sum() for i in b]);D=box_dimension(mask)
    fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].imshow(mask);axes[0].axis('off');axes[1].loglog(b,c,'o-',label=f'D={D:.3f}');axes[1].loglog(b,c[0]*(b/b[0])**(-91/48),'--',label='91/48');axes[1].legend();save(fig,out,'fractal_cluster')
    from scipy.ndimage import gaussian_filter,label
    if urban_grid is None:
        y,x=np.mgrid[:200,:300];grid=100*np.exp(gaussian_filter(rng.normal(size=x.shape),3)*8)
        for a,b in rng.uniform([0,0],[300,200],(30,2)):grid+=rng.uniform(500,3000)*np.exp(-((x-a)**2+(y-b)**2)/rng.uniform(30,300))
        grid[:,:20]=-1;kind='synthetic'
    else:grid=np.loadtxt(urban_grid);kind='provided'
    thresholds=np.linspace(0,3000,61);a=urban_statistics(grid,thresholds);table(out,f'urban_{kind}',a,'threshold,occupied_fraction,clusters,largest_fraction,mean_area,vertical_span,horizontal_span')
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for ax,t in zip(axes[0],[100,500,1000]):
        z,_=label((grid>=0)&(grid>t));ax.imshow(z,cmap='nipy_spectral');ax.set_title(f'{kind}: threshold {t}');ax.axis('off')
    for ax,col,name in zip(axes[1],[3,2,1],['Largest fraction','Clusters','Occupied fraction']):ax.plot(thresholds,a[:,col]);ax.set(xlabel='Density threshold',ylabel=name)
    save(fig,out,f'urban_{kind}')

def avalanche_study(out,rng,quick):
    n=50 if quick else 300;fig,axes=plt.subplots(3,3,figsize=(12,10));summaries=[]
    for row,R in enumerate((1.4,.9,.7)):
        h=rng.normal(0,R,(n,n));f,s,labels=simulate(h,return_map=True);b=benchmark(h);summaries.append([R,len(s),s.max()/n**2,b['brute'][0],b['sorted'][0]])
        axes[row,0].imshow(labels%20,cmap='tab20');axes[row,0].set_title(f'R={R}');axes[row,0].axis('off');axes[row,1].step(f,2*np.cumsum(s)/n**2-1);axes[row,1].set(xlabel='Field',ylabel='Magnetization');x,c=np.unique(s,return_counts=True);axes[row,2].loglog(x,c,'o');axes[row,2].set(xlabel='Avalanche size',ylabel='Count');table(out,f'avalanche_{R}',np.column_stack((f,s)),'field,size')
    table(out,'avalanche_benchmark',summaries,'disorder,count,largest_fraction,brute_seconds,sorted_seconds');save(fig,out,'avalanche_maps')

def chaos_study(out,quick):
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,mu in zip(axes.flat,[.2,.4,.6,.8,.9,1.]):
        for x in (.1,.3,.7):ax.plot(trajectory(logistic,x,mu,50),alpha=.8)
        ax.set(title=f'mu={mu}',xlabel='Iteration',ylabel='x')
    save(fig,out,'logistic_trajectories')
    d=separation(.9);a,b=fit_separation(d);fig,ax=plt.subplots();ax.semilogy(d,'o',ms=3);i=np.arange(len(d));mask=(d>1e-12)&(d<1e-2);ax.semilogy(i[mask],np.exp(a*i[mask]+b),label=f'Fit={a:.3f}');ax.legend();save(fig,out,'lyapunov_separation')
    fig,axes=plt.subplots(1,3,figsize=(12,4));eps=2.**np.arange(-4,-15,-1)
    for ax,mu in zip(axes,[.8,.892486418,.9]):
        t=trajectory(logistic,.1234567,mu,2**15 if quick else 2**18,10000);e,c,h=dimension_curve(t,eps);ax.semilogx(e,c,'o-',label='Capacity');ax.semilogx(e,h,'s-',label='Information');ax.set(title=f'mu={mu}',xlabel='Box size',ylabel='Local dimension');ax.legend();table(out,f'dimensions_{mu}',np.column_stack((e,c,h)),'epsilon,capacity,information')
    save(fig,out,'attractor_dimensions')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,mu in zip(axes,[1.,.9]):
        x,h=invariant_density(mu,100000 if quick else 1000000,200 if mu==1 else 2000);ax.plot(x,h,label='Numerical density')
        if mu==1:ax.plot(x,1/(np.pi*np.sqrt(x*(1-x))),label='Exact density')
        else:
            for q in critical_orbit(mu,8):ax.axvline(q,alpha=.3)
        ax.set(title=f'mu={mu}',xlabel='x',ylabel='Density');ax.legend()
    save(fig,out,'invariant_measures')
    mu=np.linspace(.8,1,800);fig,ax=plt.subplots();ax.plot(np.repeat(mu,128),np.concatenate([trajectory(logistic,.1,m,128,500) for m in mu]),',k');x=np.full(mu.shape,.5)
    for _ in range(8):x=logistic(x,mu);ax.plot(mu,x,lw=.6)
    ax.set(xlabel='mu',ylabel='Attractor');save(fig,out,'critical_boundaries')

def memory_study(out,rng):
    patterns=rng.choice([-1,1],(5,36));net=MemoryDynamics(patterns);fig,axes=plt.subplots(2,4,figsize=(11,6))
    for i,noise in enumerate([0,.2,.3,.4]):
        x=patterns[0].copy();x[rng.choice(36,int(noise*36),replace=False)]*=-1;states,e,o=net.evolve(x,rng,400);axes[0,i].imshow(x.reshape(6,6),cmap='gray');axes[0,i].set_title(f'{noise:.0%} corruption');axes[1,i].imshow(states[-1].reshape(6,6),cmap='gray');axes[1,i].set_title(f'Overlap {o[-1,0]:.2f}');table(out,f'memory_noise_{i}',np.column_stack((e,o)),'energy,'+','.join(f'overlap_{j}' for j in range(5)))
    for ax in axes.flat:ax.axis('off')
    save(fig,out,'memory_retrieval')
    fig,axes=plt.subplots(1,2,figsize=(10,4));rows=[]
    for count in [1,3,5,10,20,30]:
        p=rng.choice([-1,1],(count,36));n=MemoryDynamics(p);stable=np.mean(np.all(np.where(p@n.w==0,p,np.sign(p@n.w))==p,axis=1));rows.append([count,stable])
    a=np.array(rows);axes[0].plot(*a.T,'o-');axes[0].set(xlabel='Stored patterns',ylabel='Fixed-point fraction')
    for beta in (.01,.1,1.,2.):
        states,e=modern_history(patterns,patterns[0]+rng.normal(size=36),beta);axes[1].plot(e,label=f'beta={beta}')
    axes[1].set(xlabel='Iteration',ylabel='Modern energy');axes[1].legend();save(fig,out,'memory_capacity_temperature');table(out,'memory_capacity',a,'patterns,fixed_fraction')
    fig,axes=plt.subplots(2,4,figsize=(11,6));masked=patterns[0].reshape(6,6).copy();masked[3:,:]=0;masked[:,3:]=0
    for i,x in enumerate([masked.ravel(),-patterns[0],np.sign(patterns[0]-patterns[1]+patterns[2]),rng.choice([-1,1],36)]):
        s,e,o=net.evolve(x,rng,400);axes[0,i].imshow(x.reshape(6,6),cmap='gray',vmin=-1,vmax=1);axes[1,i].imshow(s[-1].reshape(6,6),cmap='gray',vmin=-1,vmax=1)
    for ax in axes.flat:ax.axis('off')
    save(fig,out,'memory_attractors')

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--urban-grid');p.add_argument('--output',type=Path,default=Path('results/extended'));p.add_argument('--study',choices=['all','ising','percolation','avalanches','chaos','memory'],default='all');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for name,fn in [('ising',lambda r:ising_study(a.output,r,a.quick)),('percolation',lambda r:percolation_study(a.output,r,a.quick,a.urban_grid)),('avalanches',lambda r:avalanche_study(a.output,r,a.quick)),('chaos',lambda r:chaos_study(a.output,a.quick)),('memory',lambda r:memory_study(a.output,r))]:
        if a.study in ('all',name):print(name,flush=True);fn(np.random.default_rng(42))
if __name__=='__main__':main()
