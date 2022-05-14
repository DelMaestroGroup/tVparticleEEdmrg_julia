using Random
using LinearAlgebra
using ProgressBars
using Tdvp
using MKL 
using ITensorGPU: CuDenseTensor, Spectrum, eltype, diagind, Dense, Tensor
using CUDA, CUDA.CUSOLVER, CUDA.CUBLAS
#using TimeEvoMPS

gpu = cu

"""Due to an bug in ITensorGPU, we need to overwrite this function """
function LinearAlgebra.svd(T::CuDenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
    maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
    mindim::Int = get(kwargs, :mindim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    absoluteCutoff::Bool = get(kwargs, :absoluteCutoff, false)
    doRelCutoff::Bool = get(kwargs, :doRelCutoff, true)
    fastSVD::Bool = get(kwargs, :fastSVD, false)

    #println("### Called svd for gpu tensor ###")

    aT = array(T) 
      MU, MS, MV = CUSOLVER.svd!(aT) 
    conj!(MV)   ## REMOVING THIS COMMENT SEEMS TO FIX NORMALIZATION ERROR
    P = MS .^ 2 

    #println(mindim, maxdim)
    #### ERROR PERSISTS IF WE COMMENT THIS OUT
    truncerr, docut, P = truncate!(
      P;
      mindim=mindim,
      maxdim=maxdim,
      cutoff=cutoff,
      absoluteCutoff=absoluteCutoff,
      doRelCutoff=doRelCutoff,
    ) 
    # truncerr = 0.0
    ######


    spec = Spectrum(P, truncerr)
    dS = length(P)
    if dS < length(MS)
      MU = MU[:, 1:dS]
      MS = MS[1:dS]
      MV = MV[:, 1:dS]
    end
  
    # Make the new indices to go onto U and V
    u = eltype(IndsT)(dS)
    v = eltype(IndsT)(dS)
    Uinds = IndsT((ind(T, 1), u))
    Sinds = IndsT((u, v))
    Vinds = IndsT((ind(T, 2), v))
    U = Tensor(Dense(vec(MU)), Uinds)
    Sdata = CUDA.zeros(ElT, dS * dS)
    dsi = diagind(reshape(Sdata, dS, dS), 0)
    Sdata[dsi] = MS
    S = Tensor(Dense(Sdata), Sinds)
    V = Tensor(Dense(vec(MV)), Vinds)
    return U, S, V, spec
end


function ITensorGPU.truncate!(
    P::CuVector{Float64}; kwargs...
  )::Tuple{Float64,Float64,CuVector{Float64}}
    maxdim::Int = min(get(kwargs, :maxdim, length(P)), length(P))
    mindim::Int = min(get(kwargs, :mindim, 1), maxdim)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    absoluteCutoff::Bool = get(kwargs, :absoluteCutoff, false)
    doRelCutoff::Bool = get(kwargs, :doRelCutoff, true)

    #println("### Called truncate! for gpu tensor ###")

    origm = length(P) 
    docut = 0.0
    maxP = maximum(P)
    if maxP == 0.0
      P = CUDA.zeros(Float64, 1)
      return 0.0, 0.0, P
    end
    if origm == 1
      docut = maxP / 2
      return 0.0, docut, P[1:1]
    end 
  
    
    #Zero out any negative weight
    #neg_z_f = (!signbit(x) ? x : 0.0)
    #println(P)

    rP = map(x -> !signbit(x) ? x : 0.0, P)
    global n = origm
    truncerr = 0.0
    if n > maxdim
        truncerr = sum(rP[1:(n - maxdim)])
        global n = maxdim
    end

    if absoluteCutoff
        #Test if individual prob. weights fall below cutoff
        #rather than using *sum* of discarded weights
        if minimum(rP) > cutoff
            cut_ind = origm
        else
            sub_arr = rP .- cutoff
            err_rP = sub_arr ./ abs.(sub_arr)
            flags = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
            cut_ind = CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1 
        end
        global n = min(maxdim, cut_ind)#min(maxdim, length(P) - cut_ind)
        global n = max(n, mindim)
        truncerr += sum(rP[(cut_ind + 1):end])
        #println(n)
    else
        scale = 1.0
        if doRelCutoff
            scale = sum(P)
            scale = scale > 0.0 ? scale : 1.0
        end

        #Continue truncating until *sum* of discarded probability 
        #weight reaches cutoff reached (or m==mindim)
        sub_arr = rP .+ truncerr .- cutoff * scale
        err_rP = sub_arr ./ abs.(sub_arr)
        flags = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
        if maximum(flags) == 0.0
            cut_ind = origm
        else
            cut_ind = CUDA.CUBLAS.iamax(length(err_rP), err_rP .* flags) - 1
        end
        if cut_ind > 0
            truncerr += sum(rP[(cut_ind + 1):end])
            global n =  min(maxdim, cut_ind) #min(maxdim, length(P) - cut_ind)
            global n = max(n, mindim)
            if scale == 0.0
                truncerr = 0.0
            else
                truncerr /= scale
            end
        else # all are above cutoff
            truncerr += sum(rP[1:maxdim])
            global n = min(maxdim, length(P)) # - cut_ind)
            global n = max(n, mindim)
            if scale == 0.0
                truncerr = 0.0
            else
                truncerr /= scale
            end
        end
    end 
    #println(n, " ", origm)
    #println(P)
    if n < 1
        global n = 1
    end
    if n < origm
      hP = collect(P)
      docut = (hP[n] + hP[n + 1]) / 2
      if abs(hP[n] - hP[n + 1]) < 1E-3 * hP[n]
        docut += 1E-3 * hP[n]
      end
    end
      rinds = 1:n
      rrP = P[rinds]
      #println(P)
      #println(rrP)
      #println("")
    return truncerr, docut, rrP
  end


function tV_dmrg_ee_calclation_quench_gpu(params::Dict{Symbol,Any},output_fh::FileOutputHandler,snapshot_sh::SnapshotHandler)
  ### Unpacking variables ###
    # Number of sites
    L = params[:L]
    # Number of Fermions
    N = params[:N]
    # Hopping
    t = params[:t]
    # Time evolution steps between measurements
    cSteps = params[:consec_steps]
    # Boundary conditions
    boundary = params[:boundary]
    # Size of region A
    Asize = params[:ee]
    ℓsize = div(L, 2)
    # Interaction paramers V, V' 
    # V is split into regions for eq. dmrg
    if params[:V_log]
        V_array = log_range(params[:V_start],params[:V_end],params[:V_num]) 
    else
        V_array = lin_range(params[:V_start],params[:V_end],params[:V_num]) 
    end 
    # flatten and sort V
    V_array = sort(vcat(V_array...))
    nV = length(V_array)
    # unpack other parameters
    V0 = params[:V0]
    Vp0 = params[:Vp0]
    Vp = params[:Vp]
    # Construct time array 
    dt::Float64 = params[:time_step]
    times = collect(Float64,params[:time_min]:dt:params[:time_max])

  ### Main calculation ### 
    # setup lattice
    sites = siteinds("Fermion",L; conserve_qns=true)
    if params[:time_min] > 0.0 
        ## load saved state
        psi = read(snapshot_sh,"state",params[:time_min])
        sites_dense = siteinds(psi)
        psi_inf, psi_bot_vec = construct_auxiliary_states(sites,L,N,V0)  
    else
        ## compute initial state
        sites_dense = removeqns(sites) 
        # solve equilibrium problem for t<0
        psi, psi_inf, psi_bot_vec = compute_equilibium_groundstate(sites,L,N,t,V0,Vp0,boundary)  
        # converte states to gpu cuMPS, need to remove qn conservation for this purpose (not implemented in ITensorGPU)
        # dense removes qns
        psi = dense(psi)
        replace_siteinds!(psi,sites_dense)
    end

    # converte states to gpu cuMPS, need to remove qn conservation for this purpose (not implemented in ITensorGPU)
    # dense removes qns
    psi_inf = dense(psi_inf)
    psi_bot_vec = dense.(psi_bot_vec)
    # need to reconstruct sites to the dense versions without conserve_fn 
    replace_siteinds!(psi_inf,sites_dense)
    for i = 1:length(psi_bot_vec)
        @inbounds psi_bot_vec[i] = replace_siteinds(psi_bot_vec[i],sites_dense)
    end 

    for (iV, V) in enumerate(V_array) 
        # print # V to all files
        write_str(output_fh,"# V = $(V)\n")
        println("\nEvolution $(iV)/$(nV) for V=$(V) ... ")
        # perform time evolution, entanglement calculation, and write to files (need to copy state psi as will be changed in the function)
        compute_entanglement_quench_gpu(L,N,t,V,Vp,boundary,times,dt,cSteps,sites_dense,copy(psi),psi_bot_vec,psi_inf,Asize,ℓsize,params[:spatial],output_fh,snapshot_sh;debug=params[:debug],tdvp=params[:tdvp],save_obdm=params[:obdm],first_order_trotter=params[:first_order_trotter])      
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
function compute_entanglement_quench_gpu(
    L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond, times::Vector{Float64}, dt::Float64, cSteps::Int64,
    sites::Any, psi::MPS, psi_bot_vec::Vector{MPS}, psi_inf::MPS,
    Asize::Int64,ℓsize::Int64,spatial::Bool,output_fh::FileOutputHandler, snapshot_sh::SnapshotHandler
    ;
    debug::Bool=false,
    trotter::Bool=true,
    first_order_trotter::Bool=false,
    tdvp::Bool=false,
    save_obdm::Bool=false) 

    psi = gpu(psi) 

    if ~tdvp && trotter && Vp == 0 
       trotter_gates = create_trotter_gates_gpu(sites,dt,L,N,t,V,Vp,boundary;first_order_trotter=first_order_trotter)
       #trotter_gates = cpu.(trotter_gates)
       trotter_gates = gpu.(trotter_gates)  
    elseif tdvp
        print("tdvp not implemented for gpu,  exit()")
        exit(1)
    else Vp != 0.0
        print("Vp!=0 not implemented for trotter gates exit()")
        exit(1)
        # H = create_hamiltonian(sites,L,N,t,V,Vp,boundary)
        # expiHt = exp(-1.0im*dt*H) 
    end
    times = [times; times[end]+dt]
    for (it,time) in enumerate(ProgressBar(times))
        # perform one time evolution step on psi 
        # skip first step at t=0 to cover initial state V0, Vp0
        if it > 1
            if tdvp
            #     # This is a placeholder for including tdvp once it was introduced in the julia version of ITensor.
            #     # As soon as it is there, just include it below, the rest is already setup for it.
            #     error("tdvp is currently not implemented in ITensor for julia. This is just a placeholder flag --tdvp.")
            #     #tdvp!(psi,H,dt,dt;hermitian=true)
                tdvp_step!(psi,projH,dt; hermitian=true,cutoff=1e-14,exp_tol=1e-12,krylovdim=20,maxiter=50)
            else 
                psi = my_apply_trotter1d(trotter_gates,psi,L)
                #psi = apply(trotter_gates,psi;cutoff=1e-14) #before: 1e-14, not set maxdim 
                # only pause time evolution for a measurement every cStep steps as 
                # copying between cpu and gpu is slow
                if (it-1) % cSteps != 0
                    continue
                end
            end
        end  

        # for a measurement need cpu at the moment
        psi_cpu = cpu(psi)
        # save snapshot to file when requested
        if time_for_snapshot(snapshot_sh,"state",it)
            write(snapshot_sh,"state",time,psi_cpu)
        end
        # compute entanglement and write to files 
        if save_obdm && Asize == 1 
            particle_ee, obdm = compute_particle_EE_and_obdm(psi_cpu,N)
            write(output_fh,"obdm",time,obdm)  
        else
            if Asize == 1 
                particle_ee = compute_particle_EE(psi_cpu,N)
            elseif Asize == 2
                if save_obdm
                    @warn "Skip obdm saving, currently only supported for n=1."
                end
                particle_ee = compute_particle_EE_n2(psi_cpu,N)
            end
        end
        write(output_fh,"particleEE",time,particle_ee) 

        if spatial
            spatial_ee, accessible_ee = compute_spatial_EE(copy(psi_cpu),ℓsize)
            write(output_fh,"spatialEE",time,spatial_ee)
            write(output_fh,"accessibleEE",time,accessible_ee)
        end

        # debug printing
        if debug  
            sp_psi_psiinf = dot(psi_cpu,psi_inf) 
            sp_psi_psibot = [dot(psi_cpu,psi_bot) for psi_bot in psi_bot_vec] 
            write(output_fh,"debug",V,abs(sp_psi_psiinf),abs.(sp_psi_psibot)) 
        end
        psi_cpu = nothing
    end  
end

 
function create_trotter_gates_gpu(sites::Any,dt::Float64,L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond;first_order_trotter::Bool=false) 
    # Time gates: https://itensor.github.io/ITensors.jl/dev/tutorials/MPSTimeEvolution.html
    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network 
    #
    # On the GPU it seems that fermions are implemented as hard-core bosons only and the
    # string operators are missing. We therefore need periodic boundary conditions for 
    # even N and odd N
 
    gates = ITensor[]
    for j=1:L-1
        s1 = sites[j]
        s2 = sites[j+1]
        hj = -t * op("Cdag",s1) * op("C", s2) -t * op("Cdag",s2) * op("C", s1) + V * op("N", s1) * op("N", s2)
        if first_order_trotter
            Gj = exp(-1.0im * dt * hj)
        else # second order Trotter gates by defauls
            Gj = exp(-1.0im * dt/2 * hj)
        end
        push!(gates,Gj)
    end
    if boundary == PBC
        factor = 1.0 # From comparison with ED, this factor is always 1 and not (N % 2 == 0) ? -1.0 : 1.0
        s1 = sites[L]
        s2 = sites[1]
        hN = -t * factor * op("Cdag",s1) * op("C", s2) -t * factor * op("Cdag",s2) * op("C", s1) + V * op("N", s1) * op("N", s2)
        if first_order_trotter
            GN = exp(-1.0im * dt * hN)
        else # second order Trotter gates by defauls
            GN = exp(-1.0im * dt/2 * hN)
        end
        push!(gates,GN)
    end
    if ~first_order_trotter
        append!(gates,reverse(gates))
    end

    return gates
end 

"""
    By chosing a cutoff 2e-12 (that is larger than 1e-12), internally factorize_eigen
    will be used by ITensors instead of factorization_svd. We found that factorize_svd
    regularly throws InexactErrors as the called underlying CUDA operation 'gesvdj!' uses 32-bit
    indexing and some index exeeds the typemax(Int32) range. We found so far that this is
    not the case (or not as early) when using factorize_eigen and the corresponding
    CUDA function 'heevd!'. As of writing this comment, the corresponding functions 
    are in '@CUDA/lib/cusolver/dense.jl' and called from  
    @ITensors/src/mps/abstractmps.jl:1772; product(o::ITensors.ITensor, ψ::MPS, ns::Vector{Int64}; move_sites_back::Bool, apply_dag::Bool, kwargs::Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:cutoff,), Tuple{Float64}}})
    -->
    ... movesites->_movesites->movesite ...
    -->
    @ITensors/src/decomp.jl:499; factorize(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:cutoff, :tags, :ortho), Tuple{Float64, String, String}}})
    -->
    @ITensors/src/decomp.jl:378; factorize_svd(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:cutoff, :tags, :ortho), Tuple{Float64, String, String}}})
    or
    @ITensors/src/decomp.jl:421; factorize_eigen(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:cutoff, :tags, :ortho), Tuple{Float64, String, String}}})    

    We currently do not know the exact reason for why eigen works and svd fails. There 
    are other operations called before the CUDA code in factorize_eigen compared to 
    factorize_svd which could potentially avoid the indexing problem. From just looking
    at the saved states before the error in factorize_svd occurs, it is also not apparent
    where the index larger 2^32 originates from (but maybe it is a linear index into
    a large 3d array, an offset row or column index, or even a 'bug' in ITensors.jl
    that only becomes visible for Int32 indices). 
    The error and a stacktrace are at the end of this file.
"""
function my_apply_trotter1d(gates::AbstractVector, psi::MPS, L::Int64; cutoff=2e-12) 
    
    for gate in gates
       psi = apply(gate,psi;cutoff=cutoff)
    end  
    #psi = apply(gates,psi;cutoff=cutoff)
     
    return psi 

end


"""
ERROR: LoadError: InexactError: trunc(Int32, 2147801088)
Stacktrace:
  [1] throw_inexacterror(f::Symbol, #unused#::Type{Int32}, val::Int64)
    @ Core ./boot.jl:612
  [2] checked_trunc_sint
    @ ./boot.jl:634 [inlined]
  [3] toInt32
    @ ./boot.jl:671 [inlined]
  [4] Int32
    @ ./boot.jl:761 [inlined]
  [5] convert
    @ ./number.jl:7 [inlined]
  [6] cconvert
    @ ./essentials.jl:417 [inlined]
  [7] macro expansion
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/libcusolver.jl:3212 [inlined]
  [8] macro expansion
    @ ~/.julia/packages/CUDA/KnJGx/src/pool.jl:173 [inlined]
  [9] macro expansion
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/error.jl:57 [inlined]
 [10] cusolverDnZgesvdj(handle::Ptr{Nothing}, jobz::Char, econ::Int64, m::Int64, n::Int64, A::CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, lda::Int64, S::CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}, U::CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, ldu::Int64, V::CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, ldv::Int64, work::CUDA.CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}, lwork::Int64, info::CUDA.CuArray{Int32, 1, CUDA.Mem.DeviceBuffer}, params::Ptr{Nothing})
    @ CUDA.CUSOLVER ~/.julia/packages/CUDA/KnJGx/lib/utils/call.jl:26
 [11] #2100
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/dense.jl:484 [inlined]
 [12] with_workspace(f::CUDA.CUSOLVER.var"#2100#2103"{Char, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, CUDA.CuArray{Int32, 1, CUDA.Mem.DeviceBuffer}, Base.RefValue{Ptr{Nothing}}, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, Int64, Int64, Int64}, eltyp::Type{ComplexF64}, size::CUDA.CUSOLVER.var"#bufferSize#2102"{Char, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, Base.RefValue{Ptr{Nothing}}, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}, Int64, CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}, Int64, Int64, Int64}, fallback::Nothing; keep::Bool)
    @ CUDA.APIUtils ~/.julia/packages/CUDA/KnJGx/lib/utils/call.jl:77
 [13] with_workspace (repeats 2 times)
    @ ~/.julia/packages/CUDA/KnJGx/lib/utils/call.jl:58 [inlined]
 [14] gesvdj!(jobz::Char, econ::Int64, A::CUDA.CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer}; tol::Float64, max_sweeps::Int64)
    @ CUDA.CUSOLVER ~/.julia/packages/CUDA/KnJGx/lib/cusolver/dense.jl:483
 [15] gesvdj!
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/dense.jl:449 [inlined]
 [16] _svd!
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/linalg.jl:114 [inlined]
 [17] #svd!#2223
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/linalg.jl:102 [inlined]
 [18] svd!
    @ ~/.julia/packages/CUDA/KnJGx/lib/cusolver/linalg.jl:102 [inlined]
 [19] svd(T::NDTensors.DenseTensor{ComplexF64, 2, Tuple{ITensors.Index{Int64}, ITensors.Index{Int64}}, NDTensors.Dense{ComplexF64, CUDA.CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}}}; kwargs::Base.Pairs{Symbol, Any, NTuple{4, Symbol}, NamedTuple{(:ortho, :cutoff, :tags, :alg), Tuple{String, Float64, String, String}}})
    @ DMRGEntanglementCalculationGPU ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/src/dmrg_entanglement_quench_calculation_functions_gpu.jl:24
 [20] svd(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, NTuple{4, Symbol}, NamedTuple{(:ortho, :cutoff, :tags, :alg), Tuple{String, Float64, String, String}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/decomp.jl:109
 [21] factorize_svd(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:ortho, :cutoff, :tags), Tuple{String, Float64, String}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/decomp.jl:378
 [22] factorize(A::ITensors.ITensor, Linds::Vector{ITensors.Index{Int64}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol, Symbol}, NamedTuple{(:ortho, :cutoff, :tags), Tuple{String, Float64, String}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/decomp.jl:498
 [23] MPS(A::ITensors.ITensor, sites::Vector{Vector{ITensors.Index{Int64}}}; leftinds::ITensors.Index{Int64}, orthocenter::Int64, kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:ortho, :cutoff), Tuple{String, Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1595
 [24] setindex!(ψ::MPS, A::ITensors.ITensor, r::UnitRange{Int64}; orthocenter::Int64, perm::Vector{Int64}, kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:ortho, :cutoff), Tuple{String, Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1537
 [25] setindex!(::MPS, ::ITensors.ITensor, ::UnitRange{Int64}, ::Pair{Symbol, Any}, ::Vararg{Pair{Symbol, Any}}; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:orthocenter, :perm), Tuple{Int64, Vector{Int64}}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1548
 [26] swapbondsites(ψ::MPS, b::Int64; kwargs::Base.Pairs{Symbol, Any, Tuple{Symbol, Symbol}, NamedTuple{(:ortho, :cutoff), Tuple{String, Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1630
 [27] movesite(ψ::MPS, n1n2::Pair{Int64, Int64}; orthocenter::Int64, kwargs::Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:cutoff,), Tuple{Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1656
 [28] _movesites(ψ::MPS, ns::Vector{Int64}, ns'::Vector{Int64}; kwargs::Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:cutoff,), Tuple{Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1682
 [29] movesites(ψ::MPS, nsns'::Vector{Pair{Int64, Int64}}; kwargs::Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:cutoff,), Tuple{Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1702
 [30] product(o::ITensors.ITensor, ψ::MPS, ns::Vector{Int64}; move_sites_back::Bool, apply_dag::Bool, kwargs::Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:cutoff,), Tuple{Float64}}})
    @ ITensors ~/.julia/packages/ITensors/ZMKMP/src/mps/abstractmps.jl:1775
 [31] my_apply_trotter1d(gates::Vector{ITensors.ITensor}, psi::MPS, L::Int64; cutoff::Float64)
    @ DMRGEntanglementCalculationGPU ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/src/dmrg_entanglement_quench_calculation_functions_gpu.jl:376
 [32] my_apply_trotter1d(gates::Vector{ITensors.ITensor}, psi::MPS, L::Int64)
    @ DMRGEntanglementCalculationGPU ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/src/dmrg_entanglement_quench_calculation_functions_gpu.jl:375
 [33] compute_entanglement_quench_gpu(L::Int64, N::Int64, t::Float64, V::Float64, Vp::Float64, boundary::BdryCond, times::Vector{Float64}, dt::Float64, cSteps::Int64, sites::Vector{ITensors.Index{Int64}}, psi::MPS, psi_bot_vec::Vector{MPS}, psi_inf::MPS, Asize::Int64, ℓsize::Int64, spatial::Bool, output_fh::FileOutputHandler, snapshot_sh::SnapshotHandler; debug::Bool, trotter::Bool, first_order_trotter::Bool, tdvp::Bool, save_obdm::Bool)
    @ DMRGEntanglementCalculationGPU ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/src/dmrg_entanglement_quench_calculation_functions_gpu.jl:294
 [34] tV_dmrg_ee_calclation_quench_gpu(params::Dict{Symbol, Any}, output_fh::FileOutputHandler, snapshot_sh::SnapshotHandler)
    @ DMRGEntanglementCalculationGPU ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/src/dmrg_entanglement_quench_calculation_functions_gpu.jl:242
 [35] main()
    @ Main ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/tVparticleEEdmrg_quench_gpu.jl:384
 [36] top-level scope
    @ ~/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/tVparticleEEdmrg_quench_gpu.jl:395
in expression starting at /nfs/home/mthamm/code/github/tVparticleEEdmrg_julia/tVparticleEEdmrg/tVparticleEEdmrg_quench_gpu.jl:394 
"""