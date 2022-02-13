
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
    # Split V in regions <0 and >0 and start from values closest to 0
    if params[:logspace]
        V_array = log_V_range(params[:V_start],params[:V_end],params[:V_num]) 
    else
        V_array = lin_V_range(params[:V_start],params[:V_end],params[:V_num]) 
    end 
    Vp = params[:Vp]

  ### Main calculation ### 
    # setup lattice
    sites = siteinds("Fermion",L; conserve_qns=true)
    # run over negative and positive V range separately
    # first V determines orthogonal states for the whole range
    # wave functions of consecutive steps are reused
    for (i,Vs) in enumerate(V_array)
        if length(Vs)> 0
            println("Calculation ",i,"/",length(V_array)," form ",Vs[1] ," to ", Vs[end] ,"...")
            psi, psi_bot_vec = create_initial_state(sites,L,N,Vs[1])
            for V in ProgressBar(Vs)
                psi, particle_ee, spatial_ee, accessible_ee = compute_dmrg_entanglement_equilibrium(L,N,t,V,Vp,boundary,sites,psi,psi_bot_vec,Asize,ℓsize,params[:spatial])

                write(output_fh,"particleEE",V,particle_ee)
                if params[:spatial]
                    write(output_fh,"spatialEE",V,spatial_ee)
                    write(output_fh,"accessibleEE",V,accessible_ee)
                end
            end
        end
    end

    return nothing
end

"""
Use initial state psi and psi_bot_vec to perform the dmrg step. Then compute entanglement calculation.

Returns:
    psi: dmrg ground state solution
    particle_EE: particle entanglement entropies vector (1 van neumann, 2-10 Renyi, 11 negativity)
    spatial_EE: all zeros if spatial=false, else spatial entanglement entropies vector
"""
function compute_dmrg_entanglement_equilibrium(
    L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond,
    sites::Vector{Index{Vector{Pair{QN, Int64}}}}, psi::MPS, psi_bot_vec::Vector{MPS},
    Asize::Int64,ℓsize::Int64,spatial::Bool)

    H = create_hamiltonian(sites,L,N,t,V,Vp,boundary)

    # dmrg parameters
    sweeps = Sweeps(10)
    maxdim!(sweeps,10,20,100,100,200)
    cutoff!(sweeps,1e-10) 
    noise!(sweeps,1E-7,1E-8,0.0) 

    # dmrg steps TODO: outputlevel set from commandline
    psi = 1/norm(psi)*psi 
    energy, psi = dmrg(H,psi_bot_vec,psi,sweeps;outputlevel=0)
    psi = 1/norm(psi)*psi 

    # compute particle entanglement entropy
    particle_EE = compute_particle_EE(copy(psi),Asize,N)


    if spatial
        # compute spatial entanglement entropy
        spatial_EE, accessible_EE = compute_spatial_EE(copy(psi),ℓsize)

        return psi, particle_EE, spatial_EE, accessible_EE
    end
    return    psi, particle_EE, zeros(Float64,size(particle_EE)), zeros(Float64,size(accessible_EE))
end

"""
Constructs new initial psi0 and psi_bot and uses these to perform dmrg step. Then compute entanglement calculation.

Returns: 
    particle_EE: particle entanglement entropies vector (1 van neumann, 2-10 Renyi, 11 negativity)
    spatial_EE: all zeros if spatial=false, else spatial entanglement entropies vector
"""
function compute_dmrg_entanglement_equilibrium(
            L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,
            boundary::BdryCond,Asize::Int64,ℓsize::Int64,
            spatial::Bool)

    sites = siteinds("Fermion",L; conserve_qns=true)
    H = create_hamiltonian(sites,L,N,t,V,Vp,boundary)
    psi, psi_bot_vec = create_initial_state(sites,L,N,V)

    # dmrg parameters
    sweeps = Sweeps(10)
    maxdim!(sweeps,10,20,100,100,200)
    cutoff!(sweeps,1e-10)
    noise!(sweeps,1E-7,1E-8,0.0) 

    # dmrg steps TODO: outputlevel set from commandline
    psi = 1/norm(psi)*psi 
    energy, psi = dmrg(H,psi_bot_vec,psi,sweeps;outputlevel=0)
    psi = 1/norm(psi)*psi 

    # compute particle entanglement entropy
    particle_EE = compute_particle_EE(psi,Asize,N)


    if spatial
        # compute spatial entanglement entropy
        spatial_EE, accessible_EE = compute_spatial_EE(psi,ℓsize)

        return particle_EE, spatial_EE, accessible_EE
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


"""
create_initial_state(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,V::Float64)
sets up the initial state depending on the t-V model phase. 

different cases:
    - V>0:
        |psi0> = random state - proj(random state->|01010101...>-|101010101...>) + |01010101...>+|101010101...>
      if also V>1:
        decrease influence of random state by 1/V^2 compared to the infinite limit
    - V<0:
        let |Psi_bot> = FT[|1110000..>, |0111000...>, |00111000...>, ...] where FT is the real Fourier transform
        with sin and cos coeficients except for the q=0 coefficent
        |psi0> =  random state - proj(random state->|Psi_bot>) +  |1110000..> +  |0111000...> + |00111000...> + ...
      if also V<-1:
        decrease influence by 1/V^2 compared to the infinite limit
"""
function create_initial_state(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,V::Float64)
    psi0 = MPS() 

    # random state : start with 0101010101 and shuffle it
    state = Vector{String}([Bool(i%2) ? "Emp" : "Occ" for i in 1:L ])
    numRands = 5*L
    for iState = 0:numRands
        shuffle!(state)
        if iState == 0
            psi0 = MPS(sites,state)
        else
            psi0 = psi0 + MPS(sites,state)
        end
    end 
    psi0 = 1/norm(psi0)*psi0
    # find states in orthogonal subspace and V->inf limit state
    psi_inf, psi_bot_vec = construct_auxiliary_states(sites,L,N,V) 
    # subtract projection onto orthogonal state 
    for psi_bot in psi_bot_vec 
        psi0 = psi0 - dot(psi_bot,psi0)*psi_bot 
    end
    # add infinite state
    if abs(V) > 1
        # towards the phase transition points, reduce influence of random state
        psi0 = 1/V^2*psi0 + psi_inf
    else
        psi0 = psi0 + psi_inf
    end
 
    return psi0, psi_bot_vec
end 


function construct_auxiliary_states(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,V::Float64)
    psi_inf = MPS()  

    if V > 0.0
        # if V positive, repulsion -> 101010... and 010101 ... as initial state
        ####
        # get all relevant state vectors
        state_vecs = Vector{Vector{String}}(undef,2)
        for iState = 0:1
            state_vecs[iState+1] = Vector{String}(undef,L)
            for iSite in 1:L
                state_vecs[iState+1][iSite] = xor(Bool(iState), Bool(iSite%2)) ? "Occ" : "Emp"
            end
        end
        # construct orthogonal and V->inf state
        psi_bot_vec = [MPS(sites,state_vecs[2]) - MPS(sites,state_vecs[1])]
        psi_inf = MPS(sites,state_vecs[2]) + MPS(sites,state_vecs[1])
    else
        # for negatve V, attractive, add states 111000000, 01110000, 001110000...
        ####
        psi_bot_vec = Vector{MPS}(undef,L-1)
        # get all relevant state vectors
        state_vecs = Vector{Vector{String}}(undef,L)
        for iState = 0:L-1
            state_vecs[iState+1] = Vector{String}(["Emp" for i in 1:L])
            for iSite in iState:(iState+N-1)
                state_vecs[iState+1][iSite%L + 1] = "Occ"
            end  
        end 
        # contruct MPS for all state_vecs
        state_mps = Vector{MPS}(undef,L)
        for (i, state_v) in enumerate(state_vecs)
            @inbounds state_mps[i]  = MPS(sites,state_v)
        end
        state_vecs = nothing
        # construct orthogonal and V->inf state
        # PROBLEM?: for save addition we start from a term that has always non-zero coefficent
        # in the sin terms we therefore use (n+1) and start with the second term
        for q = 1:N
            psi_bot_vec[q]  = 1.0*state_mps[1] 
            if q < N
                psi_bot_vec[q+N] = sin(2*pi*q/L)*state_mps[2]
            end
            for n = 1:L-1 
                psi_bot_vec[q] += cos(2*pi*q*n/L)*state_mps[n+1] 
                if n < L-1 && q<N
                    psi_bot_vec[q+N] += sin(2*pi*q*(n+1)/L)*state_mps[n+2]
                end
            end
        end 
        
        psi_inf = state_mps[1]
        for i = 2:L 
            psi_inf = psi_inf + state_mps[i]
        end

    end

    # normalize states
    for i = 1:length(psi_bot_vec)
        @inbounds psi_bot_vec[i] = 1/norm(psi_bot_vec[i])*psi_bot_vec[i]
    end
    psi_inf = 1/norm(psi_inf)*psi_inf

    # DEBUG: check that psi_inf orthogonal to every  state in psi_bot_vec
    for (i, psi_bot) in enumerate(psi_bot_vec)
        scprod = dot(psi_inf, psi_bot)
        if scprod>1e-12
            println("___________________________________________________________________")
            println("WARNING: a vector ",i,"/",length(psi_bot_vec)," in psi_bot_vec is not orthogonal to psi_inf. <A|B> =",scprod) 
            println("___________________________________________________________________")
        end
    end
    # DEBUG: compute overlap matrix of the psi_bot_vec
    # println("DEBUG: overlap matrix of bot states: only print terms >1e-14 and i!=j")
    # for i = 1:length(psi_bot_vec) 
    #     for j = 1:length(psi_bot_vec)
    #         overlap = dot(psi_bot_vec[i],psi_bot_vec[j])
    #         if overlap > 1e-14 && i!=j
    #             println("i=$(i), j=$(j), <i|j>=$(dot(psi_bot_vec[i],psi_bot_vec[j]))")
    #         end
    #     end
    # end
    # exit(0)

    return psi_inf, psi_bot_vec
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

    #S = nothing
 
    λs = λs[λs.>1e-12]
     
    spatial_EE = Vector{Float64}(undef,11)
    for α = 1:11
        spatial_EE[α] = compute_Renyi(α,λs)
    end
#_________________________________________

    αs=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,0.5]
    accessible_EE = Vector{Float64}(undef,11)
    nb =nblocks(S)[1]
    accessible_pnα =zeros(Float64,11,nb)
    counter=0;
    for i = 1:nb
        for j = 1:dims(S[Block(i, i)])[1]
            counter+=1
            for iα = 1:length(αs)
                accessible_pnα[iα,i]+=S[counter, counter]^(2*αs[iα])
            end
        end
    end
    for iα = 1:length(αs)
        accessible_pnα[iα,:]/=sum(accessible_pnα[iα,:])
    end
    for α = 1:11
        accessible_EE[α] =spatial_EE[α]- compute_InvRenyi( α,accessible_pnα[α,:])
    end



    return spatial_EE, accessible_EE
    







end

function compute_particle_EE(psi::MPS,Asize::Int64,N::Int64)
    lnN = log(N)

    # compute one body density matrix 
    obdm = correlation_matrix(psi,"Cdag","C")

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
    sumeval = sum(λs)
    if abs(sumeval-1.0)>1e-8
        println("___________________________________________________________________")
        println("WARNING: Density matrix not normalized. Sum of eigenvalues is ",sumeval,".")
        println("___________________________________________________________________")
    end 
    
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

function compute_InvRenyi(α::Int64,λs::Vector{Float64},offset::Float64 = 0.0) 
    sumeval = sum(λs)
    if abs(sumeval-1.0)>1e-8
        println("___________________________________________________________________")
        println("WARNING: Density matrix not normalized. Sum of eigenvalues is ",sumeval,".")
        println("___________________________________________________________________")
    end 
    
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
        #1/α = 2
        return -log(sum(  λs.^2)) - offset
    end

    return 1/(1-(1.0/α)) * log(sum( λs.^(1.0/α) )) - offset

end

