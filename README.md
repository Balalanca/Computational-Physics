# Computational Physics and Complex Systems

[![Tests](https://github.com/Balalanca/Computational-Physics/actions/workflows/tests.yml/badge.svg)](https://github.com/Balalanca/Computational-Physics/actions)

Numerical investigations of equilibrium and non-equilibrium statistical mechanics, critical phenomena, nonlinear dynamics, and associative memory.

## Statistical mechanics

The ferromagnetic Ising Hamiltonian is

$$H=-J\sum_{\langle i,j\rangle}s_i s_j-h\sum_i s_i.$$

Metropolis updates, incremental energy measurements, periodic spin clusters and temperature sweeps are used to investigate the approach to equilibrium. At zero field the infinite square lattice has critical temperature $T_c/J=2/\log(1+\sqrt{2})$ in units with $k_B=1$. The numerical results are compared against the exact spontaneous magnetization curve. Finite chains and finite lattices limit equilibration near criticality.

![Magnetization](results/extended/magnetization.png)
![Energy evolution](results/extended/energy_evolution.png)
![Cluster statistics](results/extended/ising_clusters.png)

Temperature quenches examine domain coarsening through cluster area, its square root as a linear-size proxy, and domain-wall bond counts. The latter are related to excess energy through $E-E_0=2J N_{\rm wall}$ for zero field. These proxies are not exact correlation lengths, and thermal excitations contribute to the wall count. The Schelling model investigates segregation through local satisfaction thresholds and vacancy-conserving relocation.

![Quench](results/extended/quench_snapshots.png)
![Coarsening](results/extended/coarsening.png)
![Segregation](results/extended/segregation.png)

## Percolation and critical phenomena

Site and bond percolation on square lattices have thresholds near 0.59274 and exactly 0.5, respectively. Cluster-size histograms are compared with the reference exponent $\tau=187/91$. The order parameter, defined here as largest-cluster area divided by lattice area, is studied with $\beta=5/36$ and finite-size exponent $\nu=4/3$.

$$P(p,L)=L^{-\beta/\nu}F((p-p_c)L^{1/\nu}).$$

![Finite-size analysis](results/extended/finite_size_scaling.png)
![Cluster configurations](results/extended/cluster_configurations.png)
![Order parameter](results/extended/supercritical_order.png)

Box counting estimates the largest-cluster fractal dimension against the theoretical value $91/48$. The synthetic population-density field illustrates thresholded urban clustering without claiming to represent an actual geographical region. A compatible geographical grid can be analysed with the same methods.

![Fractal geometry](results/extended/fractal_cluster.png)
![Synthetic urban field](results/extended/urban_synthetic.png)
![Synthetic urban cluster distributions](results/extended/urban_synthetic_clusters.png)

## Disordered systems and avalanches

The random-field Ising model evolves at zero temperature under a quasistatically increasing external field. Brute-force and sorted-list triggering methods are compared on matched disorder realizations. Spatial event labels, magnetization, size distributions and timings are recorded. Agreement of fields and sizes is checked automatically.

![Avalanche maps](results/extended/avalanche_maps.png)

## Nonlinear dynamics

The logistic map $x_{n+1}=4\mu x_n(1-x_n)$ is explored through trajectories, bifurcations, nearby-orbit separation, Lyapunov exponents, and information and capacity dimensions near the Feigenbaum point. At $\mu=1$, the invariant density is $\rho(x)=1/(\pi\sqrt{x(1-x)})$. Critical-point iterates reveal boundaries in the chaotic attractor.

![Lyapunov separation](results/extended/lyapunov_separation.png)
![Fractal dimensions](results/extended/attractor_dimensions.png)
![Invariant measures](results/extended/invariant_measures.png)
![Critical boundaries](results/extended/critical_boundaries.png)

## Associative memory

Classical Hopfield dynamics use symmetric zero-diagonal Hebbian weights and a deterministic tie rule. Asynchronous and synchronous updates, cosine overlaps, energy histories, partial inputs, negative and mixed states, noisy retrieval and fixed-point capacity are investigated. Modern associative memory uses softmax-weighted continuous retrieval and the corresponding energy landscape.

![Retrieval](results/extended/memory_retrieval.png)
![Capacity and temperature](results/extended/memory_capacity_temperature.png)
![Digit retrieval](results/extended/digit_memory.png)
![Noise sweep](results/extended/digit_noise_sweep.png)

## Reproduce

    pip install -r requirements.txt
    python -m pytest -q
    python restored_experiments.py --quick
    python additional_experiments.py

Remove `--quick` for larger lattices and longer trajectories. `--study` selects a single investigation. Provide `--urban-grid path/to/grid.txt` for a real spatial density matrix (negative entries mark excluded cells). The included extended results use the quick configuration plus the additional experiments; these are finite-size exploratory estimates, not precision measurements of universal exponents. Numerical tables are in `results/extended/`. Timing results depend on the machine. The `experiment.py` runner provides a compact baseline experiment.
