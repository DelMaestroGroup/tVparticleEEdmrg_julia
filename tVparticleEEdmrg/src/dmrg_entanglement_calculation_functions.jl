
#using OutputFileHandler 
using ITensors
using Random
using LinearAlgebra
using ProgressBars
using Combinatorics

function tV_dmrg_ee_calclation_equilibrium(params::Dict{Symbol,Any}, output_fh::FileOutputHandler)
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
    if ~(params[:V_list] === nothing) && length(params[:V_list]) > 0
        s_V = sort(params[:V_list])
        V_array = [s_V[s_V.<-2.0], s_V[-2.0 .<= s_V .< 0.0], s_V[s_V.>=0.0]]
    elseif params[:logspace]
        V_array = log_range(params[:V_start], params[:V_end], params[:V_num])
    else
        V_array = lin_range(params[:V_start], params[:V_end], params[:V_num])
    end
    Vp = params[:Vp]

    ### Main calculation ### 
    # setup lattice
    sites = siteinds("Fermion", L; conserve_qns=true)
    # run over negative and positive V range separately
    # first V determines orthogonal states for the whole range
    # wave functions of consecutive steps are reused
    for (i, Vs) in enumerate(V_array)
        if length(Vs) > 0
            println("Calculation ", i, "/", length(V_array), " form ", Vs[1], " to ", Vs[end], "...")
            psi, psi_bot_vec, psi_inf = create_initial_state(sites, L, N, Vs[1])
            for V in ProgressBar(Vs)
                psi, particle_ee, spatial_ee, accessible_ee = compute_dmrg_entanglement_equilibrium(L, N, t, V, Vp, boundary, sites, psi, psi_bot_vec, psi_inf, Asize, ℓsize, params[:spatial]; debug=params[:debug], save_obdm=params[:obdm], output_fh=output_fh)

                write(output_fh, "particleEE", V, particle_ee)
                if params[:spatial]
                    write(output_fh, "spatialEE", V, spatial_ee)
                    write(output_fh, "accessibleEE", V, accessible_ee)
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
    L::Int64, N::Int64, t::Float64, V::Float64, Vp::Float64, boundary::BdryCond,
    sites::Vector{Index{Vector{Pair{QN,Int64}}}}, psi::MPS, psi_bot_vec::Vector{MPS}, psi_inf::MPS,
    Asize::Int64, ℓsize::Int64, spatial::Bool; debug::Bool=false, save_obdm::Bool=false, output_fh::Union{IO,AbstractString,FileOutputHandler}=stdout)

    H = create_hamiltonian(sites, L, N, t, V, Vp, boundary)

    # dmrg parameters
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 100, 200, 400, 800, 1600)
    setcutoff!(sweeps, 1e-12)
    setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)

    # dmrg steps TODO: outputlevel set from commandline
    psi = 1 / norm(psi) * psi
    energy, psi = dmrg(H, psi_bot_vec, psi, sweeps; outputlevel=0)
    psi = 1 / norm(psi) * psi

    # debug printing
    if debug
        sp_psi_psiinf = dot(psi, psi_inf)
        sp_psi_psibot = [dot(psi, psi_bot) for psi_bot in psi_bot_vec]
        if isa(output_fh, FileOutputHandler)
            write(output_fh, "debug", V, energy, sp_psi_psiinf, sp_psi_psibot)
        else
            write(output_fh, "debug: V = $(V), energy = $(energy), <psi|psi_inf> = $(sp_psi_psiinf), <psi|psi_bots> = $(sp_psi_psibot) \n")
        end
    end

    # compute particle entanglement entropy
    if save_obdm && isa(output_fh, FileOutputHandler) && Asize == 1
        particle_EE, obdm = compute_particle_EE_and_obdm(copy(psi), N)
        write(output_fh, "obdm", V, obdm)
    else
        if Asize == 1
            particle_EE = compute_particle_EE(copy(psi), N)
        elseif Asize == 2
            if save_obdm
                @warn "Skip obdm saving, currently only supported for n=1."
            end
            particle_EE = compute_particle_EE_n2(copy(psi), N)
        else
            if save_obdm
                @warn "Skip obdm saving, currently only supported for n=1."
            end
            particle_EE = compute_particle_EE_n(copy(psi), N, Asize)
        end
    end


    if spatial
        # compute spatial entanglement entropy
        spatial_EE, accessible_EE = compute_spatial_EE(copy(psi), ℓsize)

        return psi, particle_EE, spatial_EE, accessible_EE
    end
    return psi, particle_EE, zeros(Float64, size(particle_EE)), zeros(Float64, size(particle_EE))
end

"""
Constructs new initial psi0 and psi_bot and uses these to perform dmrg step. Then compute entanglement calculation.
Currently only implemented for Asize=1.

Returns: 
    particle_EE: particle entanglement entropies vector (1 van neumann, 2-10 Renyi, 11 negativity)
    spatial_EE: all zeros if spatial=false, else spatial entanglement entropies vector
"""
function compute_dmrg_entanglement_equilibrium(
    L::Int64, N::Int64, t::Float64, V::Float64, Vp::Float64,
    boundary::BdryCond, Asize::Int64, ℓsize::Int64,
    spatial::Bool;
    debug::Bool=false,
    output_fh::Union{IO,AbstractString,FileOutputHandler}=stdout)

    sites = siteinds("Fermion", L; conserve_qns=true)
    H = create_hamiltonian(sites, L, N, t, V, Vp, boundary)
    psi, psi_bot_vec = create_initial_state(sites, L, N, V)

    # dmrg parameters
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 100, 200, 400, 800, 1600)
    setcutoff!(sweeps, 1e-12)
    setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)

    # dmrg steps TODO: outputlevel set from commandline
    psi = 1 / norm(psi) * psi
    energy, psi = dmrg(H, psi_bot_vec, psi, sweeps; outputlevel=0)
    psi = 1 / norm(psi) * psi

    # debug printing
    if debug
        sp_psi_psiinf = dot(psi, psi_inf)
        sp_psi_psibot = [dot(psi, psi_bot) for psi_bot in psi_bot_vec]
        if isa(output_fh, FileOutputHandler)
            write(output_fh, "debug", V, energy, sp_psi_psiinf, sp_psi_psibot)
        else
            write(output_fh, "debug: V = $(V), energy = $(energy), <psi|psi_inf> = $(sp_psi_psiinf), <psi|psi_bots> = $(sp_psi_psibot) \n")
        end
    end

    # compute particle entanglement entropy
    particle_EE = compute_particle_EE(psi, N)


    if spatial
        # compute spatial entanglement entropy
        spatial_EE, accessible_EE = compute_spatial_EE(psi, ℓsize)

        return particle_EE, spatial_EE, accessible_EE
    end
    return particle_EE, zeros(Float64, size(particle_EE)), zeros(Float64, size(accessible_EE))
end

function create_hamiltonian(sites::AbstractVector, L::Int64, N::Int64, t::Float64, V::Float64, Vp::Float64, boundary::BdryCond)

    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network 
    ampo = OpSum()
    for j = 1:L-1
        ampo += -t, "Cdag", j, "C", j + 1
        ampo += -t, "Cdag", j + 1, "C", j
        ampo += V, "N", j + 1, "N", j
        if Vp != 0.0 && j < L - 1
            ampo += Vp, "N", j + 2, "N", j
        end
    end
    if boundary == PBC
        factor = (L / 2 % 2 == 0) ? -1 : 1
        ampo += -t * factor, "Cdag", L, "C", 1
        ampo += -t * factor, "Cdag", 1, "C", L
        ampo += V, "N", 1, "N", L
        if Vp != 0.0
            ampo += Vp, "N", 2, "N", L
            ampo += Vp, "N", 1, "N", L - 1
        end
    end

    return MPO(ampo, sites)
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
function create_initial_state(sites::Vector{Index{Vector{Pair{QN,Int64}}}}, L::Int64, N::Int64, V::Float64)
    psi0 = MPS()

    # random state : start with 0101010101 and shuffle it
    psi0 = get_random_state(sites, L, N; numRands=5 * L)
    # find states in orthogonal subspace and V->inf limit state
    psi_inf, psi_bot_vec = construct_auxiliary_states(sites, L, N, V)
    # subtract projection onto orthogonal state 
    for psi_bot in psi_bot_vec
        psi0 = psi0 - dot(psi_bot, psi0) * psi_bot
    end
    # add infinite state
    if abs(V) > 1
        # towards the phase transition points, reduce influence of random state
        psi0 = 1 / V^2 * psi0 + psi_inf
    else
        psi0 = psi0 + psi_inf
    end

    return psi0, psi_bot_vec, psi_inf
end

function get_random_state(sites::Vector{Index{Vector{Pair{QN,Int64}}}}, L::Int64, N::Int64; numRands::Int64=20)
    state = Vector{String}([Bool(i % 2) ? "Emp" : "Occ" for i in 1:L])
    for iState = 0:numRands
        shuffle!(state)
        if iState == 0
            psi0 = MPS(sites, state)
        else
            psi0 = psi0 + MPS(sites, state)
        end
    end
    psi0 = 1 / norm(psi0) * psi0
    return psi0
end

function get_random_product_state(sites::Vector{Index{Vector{Pair{QN,Int64}}}}, L::Int64, N::Int64; numRands::Int64=20)
    state = Vector{String}([Bool(i % 2) ? "Emp" : "Occ" for i in 1:L])
    psi0 = randomMPS(sites, state; linkdims=numRands)
    return 1 / norm(psi0) * psi0
end

function construct_auxiliary_states(sites::Vector{Index{Vector{Pair{QN,Int64}}}}, L::Int64, N::Int64, V::Float64)
    psi_inf = MPS()

    if V >= 0.0
        # if V positive, repulsion -> 101010... and 010101 ... as initial state
        ####
        # get all relevant state vectors
        state_vecs = Vector{Vector{String}}(undef, 2)
        for iState = 0:1
            state_vecs[iState+1] = Vector{String}(undef, L)
            for iSite in 1:L
                state_vecs[iState+1][iSite] = xor(Bool(iState), Bool(iSite % 2)) ? "Occ" : "Emp"
            end
        end
        # construct orthogonal and V->inf state
        psi_bot_vec = [MPS(sites, state_vecs[2]) - MPS(sites, state_vecs[1])]
        psi_inf = MPS(sites, state_vecs[2]) + MPS(sites, state_vecs[1])
    else
        # for negatve V, attractive, add states 111000000, 01110000, 001110000...
        ####
        psi_bot_vec = Vector{MPS}(undef, L - 1)
        # get all relevant state vectors
        state_vecs = Vector{Vector{String}}(undef, L)
        for iState = 0:L-1
            state_vecs[iState+1] = Vector{String}(["Emp" for i in 1:L])
            for iSite in iState:(iState+N-1)
                state_vecs[iState+1][iSite%L+1] = "Occ"
            end
        end
        # contruct MPS for all state_vecs
        state_mps = Vector{MPS}(undef, L)
        for (i, state_v) in enumerate(state_vecs)
            @inbounds state_mps[i] = MPS(sites, state_v)
        end
        state_vecs = nothing
        # construct orthogonal and V->inf state
        # PROBLEM?: for save addition we start from a term that has always non-zero coefficent
        # in the sin terms we therefore use (n+1) and start with the second term
        for q = 1:N
            psi_bot_vec[q] = 1.0 * state_mps[1]
            if q < N
                psi_bot_vec[q+N] = sin(2 * pi * q / L) * state_mps[2]
            end
            for n = 1:L-1
                psi_bot_vec[q] += cos(2 * pi * q * n / L) * state_mps[n+1]
                if n < L - 1 && q < N
                    psi_bot_vec[q+N] += sin(2 * pi * q * (n + 1) / L) * state_mps[n+2]
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
        @inbounds psi_bot_vec[i] = 1 / norm(psi_bot_vec[i]) * psi_bot_vec[i]
    end
    psi_inf = 1 / norm(psi_inf) * psi_inf

    # DEBUG: check that psi_inf orthogonal to every  state in psi_bot_vec
    for (i, psi_bot) in enumerate(psi_bot_vec)
        scprod = dot(psi_inf, psi_bot)
        if scprod > 1e-12
            println("___________________________________________________________________")
            println("WARNING: a vector ", i, "/", length(psi_bot_vec), " in psi_bot_vec is not orthogonal to psi_inf. <A|B> =", scprod)
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
function compute_spatial_EE(psi::MPS, lsize::Int64)

    orthogonalize!(psi, lsize)
    U, S, V = svd(psi[lsize], (linkind(psi, lsize - 1), siteind(psi, lsize)))

    nVals = dim(S, 1)
    λs = zeros(Float64, nVals)
    for i = 1:nVals
        λs[i] = S[i, i]^2
    end

    #S = nothing

    λs = λs[λs.>1e-12]

    spatial_EE = Vector{Float64}(undef, 11)
    for α = 1:11
        spatial_EE[α] = compute_Renyi(α, λs)
    end
    #_________________________________________

    αs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.5]
    accessible_EE = Vector{Float64}(undef, 11)
    nb = nblocks(S)[1]
    accessible_pnα = zeros(Float64, 11, nb)
    counter = 0
    for i = 1:nb
        for j = 1:dims(S[Block(i, i)])[1]
            counter += 1
            for iα = 1:length(αs)
                accessible_pnα[iα, i] += S[counter, counter]^(2 * αs[iα])
            end
        end
    end
    for iα = 1:length(αs)
        accessible_pnα[iα, :] /= sum(accessible_pnα[iα, :])
    end
    for α = 1:11
        accessible_EE[α] = spatial_EE[α] - compute_InvRenyi(α, accessible_pnα[α, :])
    end


    return spatial_EE, accessible_EE

end

function compute_particle_EE(psi::MPS, N::Int64)
    lnN = log(N)

    # compute one body density matrix 
    obdm = correlation_matrix(psi, "Cdag", "C")

    # get obdm spectrum
    λs = abs.(eigvals!(obdm)) / N

    # get Renyi entanglement entropies
    particle_EE = Vector{Float64}(undef, 11)
    for α = 1:11
        particle_EE[α] = compute_Renyi(α, λs, lnN)
    end

    return particle_EE

end

"""Obtain the obdm in addition. Only the middle row of the one body density
matrix is returned."""
function compute_particle_EE_and_obdm(psi::MPS, N::Int64)
    lnN = log(N)

    # compute one body density matrix 
    obdm = correlation_matrix(psi, "Cdag", "C")
    #println("\n\n",reduce((x,y) -> max.(x,y), imag(obdm[N,:]/N)), " hermitian?: ", ishermitian(obdm),"\n\n")

    # get obdm spectrum
    λs = abs.(eigvals(obdm)) / N

    # get Renyi entanglement entropies
    particle_EE = Vector{Float64}(undef, 11)
    for α = 1:11
        particle_EE[α] = compute_Renyi(α, λs, lnN)
    end

    return particle_EE, real.(obdm[N, :] / N)

end

function compute_Renyi(α::Int64, λs::Vector{Float64}, offset::Float64=0.0)
    sumeval = sum(λs)
    if abs(sumeval - 1.0) > 1e-8
        println("___________________________________________________________________")
        println("WARNING: Density matrix not normalized. Sum of eigenvalues is ", sumeval, ".")
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
        return 2 * log(sum(sqrt.(λs))) - offset
    end

    return 1 / (1 - α) * log(sum(λs .^ α)) - offset

end

function compute_InvRenyi(α::Int64, λs::Vector{Float64}, offset::Float64=0.0)
    sumeval = sum(λs)
    if abs(sumeval - 1.0) > 1e-8
        println("___________________________________________________________________")
        println("WARNING: Density matrix not normalized. Sum of eigenvalues is ", sumeval, ".")
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
        return -log(sum(λs .^ 2)) - offset
    end

    return 1 / (1 - (1.0 / α)) * log(sum(λs .^ (1.0 / α))) - offset

end


function compute_particle_EE_n2(psi::MPS, N::Int64)
    lnN = log(binomial(N, 2))

    # compute two body density matrix 
    # path = String(@sprintf "bench_construction_n%02d_N%02d_%s.bench" 2 N "V-1.75")
    # Benchmark.@benchmark path " \n" 12345 begin
    #     tbdm_ijkl = correlation_matrix_n2(psi, N)
    #     tbdm = reshape_tbdm(tbdm_ijkl, N)
    # end
    tbdm_ijkl = correlation_matrix_n2(psi, N)
    tbdm = reshape_tbdm(tbdm_ijkl, N)

    λs = abs.(eigvals!(tbdm))
    λs /= sum(λs) #(N*(N-1))

    # get Renyi entanglement entropies
    particle_EE = Vector{Float64}(undef, 11)
    for α = 1:11
        particle_EE[α] = compute_Renyi(α, λs, lnN)
    end

    return particle_EE

end

function compute_particle_EE_n(psi::MPS, N::Int64, Asize::Int64)
    lnN = log(binomial(N, Asize))

    nbdm_ij = correlation_matrix_n(psi, N, Asize)
    nbdm = reshape_nbdm(nbdm_ij, N, Asize)

    λs = abs.(eigvals!(nbdm))
    λs /= sum(λs)

    # get Renyi entanglement entropies
    particle_EE = Vector{Float64}(undef, 11)
    for α = 1:11
        particle_EE[α] = compute_Renyi(α, λs, lnN)
    end

    return particle_EE

end


"""
Function to compute the L x L x L x L (for L=2N sites) correlation array
C_i,j,l,k = <psi| c_i^d c_j^d c_l c_k |psi> for the MPS state psi.

If verbose is true, print a ProgressBar for the outermost loop.
"""
function correlation_matrix_n2(psi::MPS, N::Int64; verbose=false)
    L = 2 * N
    tbdm_ijkl = zeros(Float64, L, L, L, L)
    # get sites from MPS
    s = siteinds(psi)
    # print progressbar if verbose
    if verbose
        iIter = ProgressBar(1:L)
    else
        iIter = 1:L
    end
    # loop over positions on the lattice, using commutation relations
    # C(i,j,l,k)=-C(j,i,l,k)=-C(i,j,k,l)=C(j,i,k,l)
    # and that the matrix element is real
    # C(i,j,l,k) = C(l,k,i,j)
    # and translational symmetry
    # C(i,j,l,k) = phase * C(i+1,j+1,l+1,k+1) where phase = +-1 depending
    # on the boundary conditions and how many operators were moved
    # across the boundary by translation
    for i in iIter
        for j = 1:i-1
            for k = 1:i
                for l = 1:k-1
                    # If i,j are not repeated in l,k and the sum
                    # i+j+k+l is odd then the element will be zero due
                    # to particle hole symmetry 
                    if isodd(i + j + k + l)
                        if i != k && i != l && j != k && j != l
                            continue
                        end
                    end
                    # check if value was already assigned meaning,
                    # this i,j,k,l is in a translational cycle that is
                    # already covered
                    # (this is the dirty, easy solution, maybe we can 
                    # come up with a more clever design of the loops)
                    if tbdm_ijkl[i, j, k, l] != 0.0
                        continue
                    end
                    # operator as AutoMPO, which is important such that
                    # ITensors uses all fermion relations and also
                    # sets orthogonality center, moves indices, etc.
                    # It does not easily work to just apply these
                    # operators one after the other to psi !
                    operator = AutoMPO()
                    operator += ("Cdag", j, "Cdag", i, "C", k, "C", l)
                    # convert to real MPO to use in inner
                    operator = MPO(operator, s)
                    # compute matrix element
                    element = inner(psi, operator, psi)

                    # phase is due to boundary conditions when translation
                    # moves operator over the edge
                    phase = +1.0
                    # just include all translations not caring about 
                    # potentialy different cycle lengths ?
                    for iT = 0:L-1
                        if (N % 2) == 0
                            if i + iT == L + 1
                                phase *= -1.0
                            end
                            if j + iT == L + 1
                                phase *= -1.0
                            end
                            if k + iT == L + 1
                                phase *= -1.0
                            end
                            if l + iT == L + 1
                                phase *= -1.0
                            end
                        end

                        # effects of adding translations T 
                        # <psi| T^d c_i^d c_j^d c_l c_k T |psi>
                        # iT times
                        i_ = (i + iT - 1) % L + 1
                        j_ = (j + iT - 1) % L + 1
                        k_ = (k + iT - 1) % L + 1
                        l_ = (l + iT - 1) % L + 1
                        # add the phase due to translation and bc
                        element_ = phase * element
                        # use C(i,j,l,k)=-C(j,i,l,k)=-C(i,j,k,l)=C(j,i,k,l)
                        # and C(i,j,l,k) = C(l,k,i,j)
                        # if (N%2)==0
                        #     cphase = 1.0
                        # else
                        cphase = -1.0
                        # end

                        for _ = 1:2

                            tbdm_ijkl[i_, j_, k_, l_] = element_
                            tbdm_ijkl[j_, i_, k_, l_] = cphase * element_
                            tbdm_ijkl[i_, j_, l_, k_] = cphase * element_
                            tbdm_ijkl[j_, i_, l_, k_] = element_

                            tbdm_ijkl[k_, l_, i_, j_] = element_
                            tbdm_ijkl[l_, k_, i_, j_] = cphase * element_
                            tbdm_ijkl[k_, l_, j_, i_] = cphase * element_
                            tbdm_ijkl[l_, k_, j_, i_] = element_

                            # Use reflection symmetry of ground state
                            i_ = L - i_ + 1
                            j_ = L - j_ + 1
                            k_ = L - k_ + 1
                            l_ = L - l_ + 1
                        end

                    end
                end
            end
        end
    end
    return tbdm_ijkl
end


"""
Function to reshape the L x L x L x L array C_i,j,l,k = <psi| c_i^d c_j^d c_l c_k |psi>
to the tbdm matrix of shape L^2 x L^2. Here no normalization is performed yet. The
normalization factor N*(N-1) has to be applied elsewhere.  
The basis has the form |i,k> were 1<= i<=L is the position of the first particle on the 
lattice and k the position of the second particle.
"""
function reshape_tbdm(tbdm_ijkl::Array{Float64}, N::Int64)
    L = 2 * N
    basis = Vector{Vector{Int64}}(undef, L * L - L)
    # Pauli exclusion principle prevents |i,i>
    i = 1
    for k = 1:L-1
        for j = 1:L
            basis[i] = [j, (j + k - 1) % L + 1]
            i += 1
        end
    end

    tbdm = zeros(Float64, length(basis), length(basis))

    for (i1, bs1) in enumerate(basis)
        for (i2, bs2) in enumerate(basis)
            tbdm[i1, i2] = tbdm_ijkl[bs1[1], bs1[2], bs2[1], bs2[2]]
        end
    end
    #display(tbdm) 
    return tbdm

end
# function reshape_tbdm(tbdm_ijkl::Array{ComplexF64},N::Int64)
#     L = 2*N
#     basis = Vector{Vector{Int64}}(undef,L*L-L)
#     tbdm = zeros(ComplexF64,L*L,L*L)
#     # Pauli exclusion principle prevents |i,i>
#     i = 1
#     for j = 1:L
#         for k = 1:L
#             if k == j
#                 continue
#             end
#             basis[i] = [j,k]
#             i += 1
#         end
#     end
#     for (i1,bs1) in enumerate(basis)
#         for (i2,bs2) in enumerate(basis)
#             tbdm[i1,i2] = tbdm_ijkl[bs2[2],bs2[1],bs1[1],bs1[2]]
#         end
#     end
#     #display(tbdm)
#     return tbdm

# end


function filter_indices(inds)
    Asize = div(length(inds), 2)
    # only "upper triangle"
    for i in 1:Asize-1
        if inds[i+1] <= inds[i] || inds[1] < inds[Asize+1] || inds[Asize+i+1] <= inds[Asize+i]
            return false
        end
    end
    return true
end

"""
Function to compute the L x ... x L x L x ... x L (for L=2N sites) correlation array
C_i1,...in,,j1,...,jn = <psi| c_i1^d ... c_in^d c_jn ... c_j1 |psi> for the MPS state psi.

If verbose is true, print a ProgressBar for the outermost loop.

This is not a very efficent implementation but can be used for all n. 
"""
function correlation_matrix_n(psi::MPS, N::Int64, Asize::Int64; verbose=true)
    L = 2 * N
    inv_parity_of_Asize = 1 + (-1) * (Asize % 2)
    nbdm_ij = zeros(Float64, repeat([L], 2 * Asize)...)
    # get sites from MPS
    s = siteinds(psi)
    # print progressbar if verbose
    # include all "loops" over full ranges 1:L which allows to write it 
    # in general for all Asize values but does not make use of the symmetries
    # to restrict loop ranges
    if verbose
        #iIter = ProgressBar(CartesianIndices(Tuple(repeat([1:L],2*Asize))))
        iIter = ProgressBar(Iterators.filter(filter_indices, CartesianIndices(Tuple(repeat([1:L], 2 * Asize)))))
    else
        #iIter = CartesianIndices(Tuple(repeat([1:L],2*Asize)))
        iIter = Iterators.filter(filter_indices, CartesianIndices(Tuple(repeat([1:L], 2 * Asize))))
    end
    for indices in iIter
        # check if value was already assigned meaning,
        # this i1..in, j1..jn is in a translational cycle that is
        # already covered
        # (this is the non-elegant, easy solution, maybe we can 
        # come up with a more clever design of the loops)
        if nbdm_ij[indices] != 0.0
            continue
        end
        i1_in_j1_jn = collect(Tuple(indices))
        # check Pauli principle (indices can only appear once in the C 
        # and only once on the Cdag operators)
        if length(unique(i1_in_j1_jn[1:Asize])) < length(i1_in_j1_jn[1:Asize])
            continue
        end
        if length(unique(i1_in_j1_jn[Asize+1:end])) < length(i1_in_j1_jn[Asize+1:end])
            continue
        end
        # particle hole symmetry 
        if (sum(i1_in_j1_jn) % 2) == inv_parity_of_Asize
            if length(unique(i1_in_j1_jn)) == length(i1_in_j1_jn)
                continue
            end
        end
        # construct the operator tuples for the AutoMPO below
        # the form is ("operator name", index, "next operator name", next_index, ...)
        cdags = Vector{Any}(repeat(["Cdag"], 2 * Asize))
        cs = Vector{Any}(repeat(["C"], 2 * Asize))
        for ind_c = 1:Asize
            cdags[2*ind_c] = i1_in_j1_jn[ind_c]
            cs[2*ind_c] = reverse(i1_in_j1_jn)[ind_c]
        end
        opr = Tuple(vcat(cdags, cs))
        # operator as AutoMPO, which is important such that
        # ITensors uses all fermion relations and also
        # sets orthogonality center, moves indices, etc.
        # It does not easily work to just apply these
        # operators one after the other to psi !
        operator = AutoMPO()
        operator += opr
        # convert to real MPO to use in inner
        operator = MPO(operator, s)
        # compute matrix element
        element = inner(psi, operator, psi)

        # phase is due to boundary conditions when translation
        # moves operator over the edge
        phase = +1.0
        # just include all translations not caring about 
        # potentialy different cycle lengths ?
        for iT = 0:L-1
            if (N % 2) == 0
                for ind_c in i1_in_j1_jn
                    if ind_c + iT == L + 1
                        phase *= -1.0
                    end
                end
            end
            # add the phase due to translation and bc
            element_ = phase * element
            # effects of adding translations T 
            # <psi| T^d c_i1^d ... c_in^d c_jn ... c_j1 T |psi>
            # iT times
            inds_ = (i1_in_j1_jn .+ iT .- 1) .% L .+ 1

            # loop for reflection symmetry
            for _ = 1:2
                # setting up all permutations of the C and Cdag operators
                # to make use of commutation relations below
                # therefore also need the parity of the permutations
                # which define the sign 
                perms_cdag = permutations(inds_[1:Asize])
                parity_cdag = (-2.0 .* parity.(sortperm.(perms_cdag)) .+ 1.0) .* (-2.0 .* parity(sortperm(inds_[1:Asize])) .+ 1.0)
                perms_c = permutations(inds_[Asize+1:end])
                parity_c = (-2.0 .* parity.(sortperm.(perms_c)) .+ 1.0) .* (-2.0 .* parity(sortperm(reverse(inds_[Asize+1:end]))) .+ 1.0)

                for (prefactor_, inds_perm_) in zip(Iterators.product(parity_cdag, parity_c), Iterators.product(perms_cdag, perms_c))
                    inds_perm = vcat(inds_perm_...)
                    prefactor = prod(prefactor_)
                    nbdm_ij[inds_perm...] = prefactor * element_
                    # use that the resulting expectation value is real
                    # i.e. <c^d_i1...c^d_in c_j1...c_jn> = <c^d_j1...c^d_jn c_i1...c_in>
                    inds_perm_T = vcat(reverse(inds_perm_)...)
                    nbdm_ij[inds_perm_T...] = prefactor * element_
                end
                # reflection symmetry
                inds_ .= L .- inds_ .+ 1
            end
        end

    end
    return nbdm_ij
end

"""
Function to reshape the L x ... x L x L x ... x L array C_i1...in,j1...jn = <psi| c_i1^d ... c_in^d c_jn ... c_j1 |psi>
to the tbdm matrix of shape L^n x L^n. Here no normalization is performed yet.  
The basis has the form |i1,...,in> were 1<= i<=L is the position of the i-th particle on the 
lattice.
"""
function reshape_nbdm(nbdm_ij::Array{Float64}, N::Int64, Asize::Int64)
    L = 2 * N
    basis = permutations(1:L, Asize)
    nbdm = zeros(Float64, length(basis), length(basis))

    for (i1, bs1) in enumerate(basis)
        for (i2, bs2) in enumerate(basis)
            nbdm[i1, i2] = nbdm_ij[bs1..., bs2...]
        end
    end

    return nbdm

end