using Random
using LinearAlgebra
using ProgressBars
using Tdvp
using MKL 
#using TimeEvoMPS

function tV_dmrg_ee_calclation_quench(params::Dict{Symbol,Any},output_fh::FileOutputHandler)
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
    times = collect(Float64,0.0:dt:params[:time_max])

  ### Main calculation ### 
    # setup lattice
    sites = siteinds("Fermion",L; conserve_qns=true)
    # solve equilibrium problem for t<0
    psi, psi_inf, psi_bot_vec = compute_equilibium_groundstate(sites,L,N,t,V0,Vp0,boundary)  

    for (iV, V) in enumerate(V_array)
        # print # V to all files
        write_str(output_fh,"# V = $(V)\n")
        println("\nEvolution $(iV)/$(nV) for V=$(V) ... ")
        # perform time evolution, entanglement calculation, and write to files (need to copy state psi as will be changed in the function)
        compute_entanglement_quench(L,N,t,V,Vp,boundary,times,dt,sites,copy(psi),psi_bot_vec,psi_inf,Asize,ℓsize,params[:spatial],output_fh;debug=params[:debug],tdvp=params[:tdvp],save_obdm=params[:obdm],first_order_trotter=params[:first_order_trotter])      
    end

    return nothing
end

function compute_equilibium_groundstate(sites::Vector{Index{Vector{Pair{QN, Int64}}}},L::Int64,N::Int64,t::Float64,V0::Float64,Vp0::Float64,boundary::BdryCond)
    # TODO analytic solution if V0==0 && Vp0==0

    # Hamiltonian before quench
    H = create_hamiltonian(sites,L,N,t,V0,Vp0,boundary)  
    # dmrg parameters
    sweeps = Sweeps(10) 
    setmaxdim!(sweeps, 100, 200, 400, 800, 1600)
    setcutoff!(sweeps, 1e-12)
    setnoise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
    # initial state
    psi, psi_bot_vec, psi_inf = create_initial_state(sites,L,N,V0) 
    # dmrg step
    energy, psi = dmrg(H,psi_bot_vec,psi,sweeps;outputlevel=0) 
    return 1/norm(psi)*psi, psi_inf, psi_bot_vec
end

"""
Use initial state psi and psi_bot_vec to perform the dmrg step. Then compute entanglement calculation.

Returns:
    psi: dmrg ground state solution
    particle_EE: particle entanglement entropies vector (1 van neumann, 2-10 Renyi, 11 negativity)
    spatial_EE: all zeros if spatial=false, else spatial entanglement entropies vector
"""
function compute_entanglement_quench(
    L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond, times::Vector{Float64}, dt::Float64,
    sites::Vector{Index{Vector{Pair{QN, Int64}}}}, psi::MPS, psi_bot_vec::Vector{MPS}, psi_inf::MPS,
    Asize::Int64,ℓsize::Int64,spatial::Bool,output_fh::FileOutputHandler
    ;
    debug::Bool=false,
    trotter::Bool=true,
    first_order_trotter::Bool=false,
    tdvp::Bool=false,
    save_obdm::Bool=false)

    if ~tdvp && trotter && Vp == 0 
       trotter_gates = create_trotter_gates(sites,dt,L,N,t,V,Vp,boundary;first_order_trotter=first_order_trotter)
    elseif tdvp
        H = create_hamiltonian(sites,L,N,t,V,Vp,boundary) 
        projH = ProjMPO(H)
        H = nothing
    else Vp != 0.0
        print("Vp!=0 not implemented for trotter gates exit()")
        exit(1)
        # H = create_hamiltonian(sites,L,N,t,V,Vp,boundary)
        # expiHt = exp(-1.0im*dt*H) 
    end

    for time in ProgressBar(times) 
        # perform one time evolution step on psi 
        # skip first step at t=0 to cover initial state V0, Vp0
        if time > 0
            if tdvp
            #     # This is a placeholder for including tdvp once it was introduced in the julia version of ITensor.
            #     # As soon as it is there, just include it below, the rest is already setup for it.
            #     error("tdvp is currently not implemented in ITensor for julia. This is just a placeholder flag --tdvp.")
            #     #tdvp!(psi,H,dt,dt;hermitian=true)
                tdvp_step!(psi,projH,dt; hermitian=true,cutoff=1e-14,exp_tol=1e-12,krylovdim=20,maxiter=50)
            else 
                psi = apply(trotter_gates,psi;cutoff=1e-14)  
            end
        end

        # compute entanglement and write to files
        if save_obdm && isa(output_fh,FileOutputHandler) && Asize == 1 
            particle_ee, obdm = compute_particle_EE_and_obdm(copy(psi),N)
            write(output_fh,"obdm",time,obdm)  
        else
            if Asize == 1 
                particle_ee = compute_particle_EE(copy(psi),N)
            elseif Asize == 2
                if save_obdm
                    @warn "Skip obdm saving, currently only supported for n=1."
                end
                particle_ee = compute_particle_EE_n2(copy(psi),N)
            end 
        end
        write(output_fh,"particleEE",time,particle_ee)  

        if spatial
            spatial_ee, accessible_ee = compute_spatial_EE(copy(psi),ℓsize)
            write(output_fh,"spatialEE",time,spatial_ee)
            write(output_fh,"accessibleEE",time,accessible_ee)
        end

        # debug printing
        if debug  
            sp_psi_psiinf = dot(psi,psi_inf) 
            sp_psi_psibot = [dot(psi,psi_bot) for psi_bot in psi_bot_vec] 
            write(output_fh,"debug",V,abs(sp_psi_psiinf),abs.(sp_psi_psibot)) 
        end
    end  
end

 
function create_trotter_gates(sites::Vector{Index{Vector{Pair{QN, Int64}}}},dt::Float64,L::Int64,N::Int64,t::Float64,V::Float64,Vp::Float64,boundary::BdryCond;first_order_trotter::Bool=false)
    # Time gates: https://itensor.github.io/ITensors.jl/dev/tutorials/MPSTimeEvolution.html
    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network 
 
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
        factor = (L/2 % 2 == 0) ? -1 : 1
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

