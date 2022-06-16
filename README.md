# tVparticleEEdmrg_julia
Particle entanglement entropy in the J-V model using DMRG with ITensor for Julia


Features:

- implements the J-V model with DMRG using [ITensors.jl](https://github.com/ITensor/ITensors.jl), [ITensorGPU.jl](https://github.com/ITensor/ITensors.jl/tree/main/ITensorGPU) [Fishman et al. (2020), [arXiv:2007.14822](https://arxiv.org/abs/2007.14822)]
- compute n particle entanglement entropy
- compute n body density matrix 
- compute spatial entanglement entropy
- equilibrium case or interaction quantum quench

Information:
- a brief description of the tricks used in the code can be found in arXiv:XXXXXX

Required julia packages:
-  Arpack
-  CUDA
-  ITensors
-  ITensorGPU
-  KrylovKit
-  MKL
-  ProgressBars
-  Combinatorics

Install:
- Clone this repository
- Install required packages; it is advised to precompile ITensors.jl

Usage: 

1. Equilibrium case
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg.jl --help` for the equilibrium code.
  
2. Interaction quench
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg_quench.jl --help` for the quench cpu code.

3. Interaction quench GPU version
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg_quench_gpu.jl --help` for the quench gpu code. The GPU is used for applying Trotter gates for time evolution to the state MPS. 
    -  The `--t-min-auto` flag allows automatic loading of the corresponding state file with largest time already present in the state file save directory (if no file is present, it starts at the quench time t=0) 
    -  The GPU code relies on ITensorGPu which currently has some issues that are fixed "by hand" in the code

Usage Example:
- Run code from within the `tVparticleEEdmrg` directory 

1. Equilibrium case (example one-particle EE for 16 fermions on 32 sites for 10 interactions between $V/J=-0.5$ and $V/J=0.5$)
   ```
   mkdir -p ./out
   julia ./tVparticleEEdmrg.jl --out "./out" --V-start -0.5 --V-end 0.5 --V-num 10 --spatial --ee 1 32 16
   ```

2. Interaction quench (example one-particle EE for 5 fermions on 10 sites, quench from free fermions to 10 different interactions between $V/J=-0.5$ and $V/J=0.5$ at $t=0$, evaluated till $t=40$ with $dt=0.01$)
   ```
   mkdir -p ./out
   julia ./tVparticleEEdmrg_quench.jl --out "./out" --V-start -0.5 --V-end 0.5 --V-num 10 --spatial --time-step 0.01 --time-max 40 --ee 1 10 5
   ```

3. Interaction quench on GPU (example one-particle EE for 14 fermions on 28 sites, quench from free fermions to interaction $V/J=0.5$ at $t=0$, evaluated till $t=40$; snapshot of the state saved every 50th time step -> automatically resuming from last saved state if one is present in `--out-states` folder)
   ```
   mkdir -p ./out
   mkdir -p ./out/states
   julia ./tVparticleEEdmrg_quench_gpu.jl --out "./out" --out-states "./out/states" --V-list 0.5 --time-max 40 --time-step 0.01 --snapshots-every 50 --time-min-auto --ee  1 28 14
   ```