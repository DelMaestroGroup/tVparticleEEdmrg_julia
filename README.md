# tVparticleEEdmrg_julia
Particle entanglement entropy in the t-V model using DMRG with ITensor for Julia


Features:

- implements the J-V model with DMRG using ITensors.jl, ITensorGPU.jl
- compute one particle entanglement entropy
- compute one body density matrix (obdm)
- compute spatial entanglement entropy

Setups: 

1. Equilibrium case
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg.jl --help` for the equilibrium code.
  
2. Interaction quench
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg_quench.jl --help` for the quench cpu code.

3. Interaction quench GPU version
    -  Use `julia ./tVparticleEEdmrg/tVparticleEEdmrg_quench_gpu.jl --help` for the quench gpu code. The GPU is used for applying Trotter gates for time evolution to the state MPS. 
    -  The `--t-min-auto` flag allows automatic loading of the corresponding state file with largest time already present in the state file save directory (if no file is present, it starts at the quench time t=0) 
    -  The GPU code relies on ITensorGPu which currently has some issues that are fixed "by hand" in the code
