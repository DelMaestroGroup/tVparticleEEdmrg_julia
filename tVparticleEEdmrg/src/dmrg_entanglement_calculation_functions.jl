
#using OutputFileHandler 
using ITensors
using Random
using LinearAlgebra
using ProgressBars

function tV_dmrg_ee_calclation_equilibrium(params::Dict{Symbol,Any},output_fh::FileOutputHandler)
  ### Unpacking variables ###
    # Number of sites
    L = params[:L]
    # Number of Fermions
    N = params[:N]
    # Hopping
    t = params[:t]
    # Boundary conditions
    boundary = params[:boundary]
    # Size of region A
    Asize = params[:ee]
    ℓsize = div(L, 2)
    # Interaction paramers V, V' 
    V_array = params[:V_start]:params[:V_step]:params[:V_end]
    Vp = params[:Vp]

  ### Main calculation ###
    for V in ProgressBar(V_array)
        particle_ee, spatial_ee = compute_dmrg_entanglement_equilibrium(L,N,t,V,Vp,boundary,Asize,ℓsize,params[:spatial])

        write(output_fh,"particleEE",V,particle_ee)
        if params[:spatial]
            write(output_fh,"spatialEE",V,spatial_ee)
        end
    end

    return nothing
end

function compute_dmrg_entanglement_equilibrium(
            L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,
            boundary::BdryCond,Asize::Int64,ℓsize::Int64,
            spatial::Bool)

    sites = siteinds("Fermion",L; conserve_qns=true)
    H = create_hamiltonian(sites,L,N,t,V,Vp,boundary)
    psi = create_initial_state(sites,L,N,V)

    # dmrg parameters
    sweeps = Sweeps(5)
    maxdim!(sweeps,10,20,100,100,200)
    cutoff!(sweeps,1e-10)
    noise!(sweeps,1e-5,1e-8)

    # dmrg steps TODO: outputlevel set from commandline
    #psi = 1/norm(psi)*psi 
    energy, psi = dmrg(H,psi,sweeps;outputlevel=0)

    # TODO: save state and energy to file

    # compute particle entanglement entropy
    particle_EE = compute_particle_EE(psi,Asize,N)


    if spatial
        # compute spatial entanglement entropy
        spatial_EE = compute_spatial_EE(psi,ℓsize)

        return particle_EE, spatial_EE
    end
    return    particle_EE, zeros(Float64,size(particle_EE))
end

function create_hamiltonian(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond)
 
    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network 
    ampo = OpSum()
    for j=1:L-1
        ampo += -t,"Cdag",j,"C",j+1
        ampo += -t,"Cdag",j+1,"C",j
        ampo += V,"N",j+1,"N",j
        if Vp != 0.0  && j < L -1
            ampo += Vp,"N",j+2,"N",j
        end
    end
    if boundary == PBC
        factor = (L/2 % 2 == 0) ? -1 : 1
        ampo += -t*factor,"Cdag",L,"C",1
        ampo += -t*factor,"Cdag",1,"C",L
        ampo += V,"N",1,"N",L
        if Vp != 0.0 
            ampo += Vp,"N",2,"N",L
            ampo += Vp,"N",1,"N",L-1
        end
    end

    return MPO(ampo,sites)
end

function create_initial_state(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,V::Float64)
    psi0 = MPS()
    if -0.5<V<1
        # in LL limit just add 2L+1 random occupations
        # start with 0101010101 and shuffle it
        state = Vector{String}([Bool(i%2) ? "Emp" : "Occ" for i in 1:L ])
        for iState = 0:2*L
            shuffle!(state)
            if iState == 0
                psi0 = MPS(sites,state)
            else
                psi0 = sum(psi0,MPS(sites,state))
            end
        end

    elseif V > 0.0
        # if V positive, repulsion -> 101010... and 010101 ... as initial state
        for iState = 0:1
            state = Vector{String}(undef,L)
            for iSite in 1:L
                state[iSite] = xor(Bool(iState), Bool(iSite%2)) ? "Occ" : "Emp"
            end
            if iState == 0
                psi0 = MPS(sites,state)
            else
                psi0 = sum(psi0,MPS(sites,state))
            end
        end
    else
        # for negatve V, attractive, add states 111000000, 01110000, 001110000...
        for iState = 0:L-1
            state = Vector{String}(["Emp" for i in 1:L])
            for iSite in iState:(iState+N-1)
                state[iSite%L + 1] = "Occ"
            end 
            if iState == 0
                psi0 = MPS(sites,state)
            else
                psi0 = sum(psi0,MPS(sites,state))
            end
        end
    end

    return psi0 
end

"""Compute Renyi spatial entanglement entropies (see ITensor documentation
https://itensor.github.io/ITensors.jl/dev/examples/MPSandMPO.html#Computing-the-Entanglement-Entropy-of-an-MPS

Returns:
    spatial_EE  : Vector{Float64}: entries:
                                        i=1     : S1 (van Neumann)
                                        i=2-10  : Si (Renyi α=i)
                                        i=11    : S1/2 (negativity entanglement)

"""
function compute_spatial_EE(psi::MPS,lsize::Int64)
    
    orthogonalize!(psi, lsize)
    U,S,V = svd(psi[lsize], (linkind(psi, lsize-1), siteind(psi,lsize)))

    nVals = dim(S,1)
    λs = zeros(Float64,nVals)
    for i = 1:nVals
        λs[i] = S[i,i]^2
    end

    S = nothing
 
    λs = λs[λs.>1e-12]
     
    spatial_EE = Vector{Float64}(undef,11)
    for α = 1:11
        spatial_EE[α] = compute_Renyi(α,λs)
    end

    return spatial_EE
    
end

function compute_particle_EE(psi::MPS,Asize::Int64,N::Int64)
    lnN = log(N)

    # compute one body density matrix
    psi = 1/norm(psi)*psi
    obdm = correlation_matrix(psi,"Cdag","C")

    # TODO: save obdm

    # get obdm spectrum
    λs = abs.(eigvals!(obdm))/N

    # get Renyi entanglement entropies
    particle_EE = Vector{Float64}(undef,11)
    for α = 1:11
        particle_EE[α] = compute_Renyi(α,λs,lnN)
    end 

    return particle_EE
    
end

function compute_Renyi(α::Int64,λs::Vector{Float64},offset::Float64 = 0.0)    
    if α == 1
        # van Neumann entropy
        See = 0.0
        for λ in λs
            if λ > 0.0
                See -= λ * log(λ)
            end
        end
        return See - offset

    elseif α == 11
        # Entanglement negativity α = 1/2
        return 2*log(sum( sqrt.(λs) )) - offset
    end

    return 1/(1-α) * log(sum( λs.^α )) - offset

end