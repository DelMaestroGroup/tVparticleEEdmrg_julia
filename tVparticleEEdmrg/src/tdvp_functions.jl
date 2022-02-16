"""
We use the code from https://github.com/orialb/TimeEvoMPS.jl (Ori Alberton) 
and adjust it to our needs.
The idea is based on 
    [1] Vidal, G. (2004). Efficient Simulation of One-Dimensional Quantum Many-Body Systems.
        Physical Review Letters, 93(4), 040502. https://doi.org/10.1103/PhysRevLett.93.040502
    [2] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
        Unifying time evolution and optimization with matrix product states. Physical Review B, 94(16).
        https://doi.org/10.1103/PhysRevB.94.165116
"""

using ITensors: position!,  orthogonalize!
using KrylovKit: exponentiate

singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

function tdvp_step!(psi::MPS,projH::ProjMPO,dt::Float64; kwargs...)

    hermitian = get(kwargs,:hermitian,true)
    exp_tol = get(kwargs,:exp_tol, 1e-14)
    krylovdim = get(kwargs,:krylovdim, 30 )
    maxiter = get(kwargs,:maxiter,100)

    N = length(psi)
    orthogonalize!(psi,1)
    position!(projH,psi,1)

    for (b,ha) in sweepnext(N)
        #evolve with two-site Hamiltonian
        twosite!(projH)
        ITensors.position!(projH,psi,b)
        wf = psi[b]*psi[b+1]
        wf, info = exponentiate(projH, -1.0im*dt/2, wf; ishermitian=hermitian , tol=exp_tol, krylovdim=krylovdim)
        dir = ha==1 ? "left" : "right"
        info.converged==0 && throw("exponentiate did not converge")
        replacebond!(psi,b,wf;normalize=false, ortho = dir, kwargs... )

        # evolve with single-site Hamiltonian backward in time. 
        i = ha==1 ? b+1 : b 
        singlesite!(projH)
        ITensors.position!(projH,psi,i)
        psi[i], info = exponentiate(projH,1.0im*dt/2,psi[i]; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim,
                                    maxiter=maxiter)
        info.converged==0 && throw("exponentiate did not converge")
    end
end 