# Entanglement entropy of the t-V-V` Model at half-filling - dependence on interaction strength V

push!(LOAD_PATH, joinpath(dirname(@__FILE__), "src"))
 
using ArgParse 
using Printf 
using ProgressBars
using Dates
using Pkg
using DMRGEntanglementCalculationGPU
using OutputFileHandler  
using Random
using Utils
using KrylovKit: exponentiate  
using ITensors.HDF5 
using ITensorGPU: MPS


# ------------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings()
    s.autofix_names = true
    @add_arg_table s begin
        "L"
            help = "number of sites"
            arg_type = Int
            required = true
        "N"
            help = "number of particles"
            arg_type = Int
            required = true
        "--out"
            metavar = "FOLDER"
            help = "path to output folder"  
        "--out-states"
            metavar = "FOLDER"
            help = "path to output folder for stat MPS snapshots, default is --out"  
        "--snapshots-every" 
            arg_type = Int
            help = "number of gate applications before a snapshot of the time evolved MPS is saved to file, 0 means never. Snapshots are only saved before measurements, therefore actual snapshop frequency will depend on --consec-steps as well."  
            default = 100
        "--spatial"
            help = "output the spatial entanglement entropy for ℓ = M/2"
            action = :store_true  
        "--obdm"
            help = "store the mid row of the obdm to file"
            action = :store_true 
        "--tdvp"
            help = "Use tdvp algorithm from https://github.com/orialb/TimeEvoMPS.jl"
            action = :store_true 
        "--first-order-trotter"
            help = "use a first order Trotter decomposition instead of 2nd order (ignored if --tvdp is set)"
            action = :store_true 
        "--no-flush"
            help = "do not flush write buffer to output files in after computation for each V" 
            action = :store_true 
        "--debug"
            help = "print <psi|psi_bot>, <psi|psi_inf>, and gs energy for each step to a file" 
            action = :store_true 
    end
    add_arg_group(s, "boundary conditions")
    @add_arg_table s begin
        "--pbc"
            help = "periodic boundary conditions (default)"
            arg_type = BdryCond
            action = :store_const
            dest_name = "boundary"
            constant = PBC
            default = PBC
        "--obc"
            help = "open boundary conditions"
            arg_type = BdryCond
            action = :store_const
            dest_name = "boundary"
            constant = OBC
            default = PBC
    end
    add_arg_group(s, "tV parameters")
    @add_arg_table s begin 
        "--V-start"
            metavar = "V_start"
            help = "start V"
            arg_type = Float64
            default = -2.0
        "--V-end"
            metavar = "V_end"
            help = "end V"
            arg_type = Float64
            default = 2.0
        "--V-num"
            metavar = "V_num"
            help = "number of V values"
            arg_type = Int64
            default = 100 
        "--V-log" 
            help = "logarithmic spacing of V values around 0"
            action = :store_true 
        "--V-list"
            metavar = "V_num"
            nargs = '*' 
            help = "multiple V values used as V_array and ignore V-start, V-end, V-step, and --V-log"
            arg_type = Float64 
            required = false
        "--Vp"
            metavar = "Vp"
            help = "final Vp"
            arg_type = Float64
            default = 0.0
        "--V0"
            metavar = "V0"
            help = "initial V"
            arg_type = Float64
            default = 0.0
        "--Vp0"
            metavar = "Vp0"
            help = "initial Vp"
            arg_type = Float64
            default = 0.0
        "--t"
            metavar = "t"
            help = "t value"
            arg_type = Float64
            default = 1.0
    end
    add_arg_group(s, "time parameters")
    @add_arg_table s begin 
        "--time-max" 
            help = "maximum time"
            arg_type = Float64
            default = 40.0
        "--time-min" 
            help = "start time, if not 0.0, a state file with a previously evolved state psi must be present."
            arg_type = Float64
            default = 0.0
        "--time-min-auto"
            help = "if set, --time-min is ignored, the start time is the one with the state file with largest time that is saved in the --out-states directory (if none use 0.0)." 
            action = :store_true 
        "--time-step" 
            help = "time step"
            arg_type = Float64
            default = 0.01 
        "--consec-steps"
            help = "consecutive time evolution steps before an entanglement measurement is performed."
            arg_type = Int64
            default = 10
    end
    add_arg_group(s, "dmrg parameter")
    @add_arg_table s begin
        "--seed"
            metavar = "seed"
            help = "seed for random number generator"
            arg_type = Int
            default = 12345
    end
    add_arg_group(s, "entanglement entropy")
    @add_arg_table s begin
        "--ee"
            metavar = "ℓ"
            help = "compute all EEs with partition size ℓ"
            arg_type = Int
            required = true
    end 

        return parse_args(s, as_symbols=true)
end



"""Runs entanglement calculation for a range of interaction strenths V based on the input parameters.
 
"""
function main()
 # _____________1_Parameter_Setup________________
    c=parse_commandline() 
    # For now only allow half filling
    if c[:L] != 2*c[:N]
        print("Not at half-filling: the number of sites L=", c[:L]," and the number of particles N=",c[:N] ," exit()")
        exit(1)
    end
    if c[:ee] != 1 && c[:ee] != 2
        print("Currently the dmrg calculation only supports one particle entanglement entropy (and two-particle entanglement, experimental)  not n=",c[:ee], " . exit()")
        exit(1)
    end
    if c[:obdm] && c[:ee] > 1
        print("Obdm can only be computed with n=1, but n=",c[:ee],". exit()")
        exit(1)
    end
    if c[:out] === nothing
        out_folder = "./"
    else
        out_folder = c[:out]
    end 
    if c[:out_states] === nothing
        states_out_folder = out_folder
    else
        states_out_folder = c[:out_states]
    end   
    
    # state file save locations
    snapshot_label = @sprintf "M%02d_N%02d_t%+5.3f_Vp%+5.3f_tstep%+5.3f_Vsta%+5.3f_Vend%+5.3f_Vnum%04d" c[:L] c[:N] c[:t] c[:Vp] c[:time_step] c[:V_start] c[:V_end] c[:V_num]
    function file_name_state(t::Float64)
        return @sprintf "state_%s_t%4.4f.dat" snapshot_label t
    end 
    if c[:time_min_auto]
        
        files_there = readdir(states_out_folder)
        for time_guess in c[:time_max]:-c[:time_step]:0.0
            path = file_name_state(time_guess)
            c[:time_min] = time_guess
            if path in files_there 
                break
            end
        end
        println("--time-min-auto found state file for t=$(c[:time_min]).")
    end

 # _____________2_Output_Setup___________________
 #
 #  Format of the files:
 #      # HEADER
 #      # start time 
 #
 #      # V_array: V0 V1 V2 V3 V4 ...
 #      # times: t0 t1 t2 t3 t4 ...
 #      # dims: ntime nV
 #  
 #      # names of cols 
 #      # V0 = V0
 #      data row V0 t0
 #      data row V0 t1
 #      ...
 #      data row V0 tntime
 #      # V1 = V1
 #      data row V1 t0
 #      ...
 #      ...
 #      data row VnV tntime
 #
 #
 #      # FOOTER
    function write_info(output_fh::FileOutputHandler,handler_name::String,c::Dict{Symbol,Any})
        if c[:V_log]
            V_array = log_range(c[:V_start],c[:V_end],c[:V_num]) 
        else
            V_array = lin_range(c[:V_start],c[:V_end],c[:V_num]) 
        end 
        # flatten and sort V
        V_array = sort(vcat(V_array...))
        times = collect(0.0:c[:time_step]:c[:time_max])  
        dims = Vector{Int64}([length(times),length(V_array)])
        write_str(output_fh,handler_name, "# start time $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
        write_str(output_fh,handler_name, @sprintf "\n# V_array: %s" join([@sprintf "%24.12E" V for V in V_array], ""))
        write_str(output_fh,handler_name, @sprintf  "\n# times: %s" join([@sprintf "%24.12E" t for t in times], ""))
        write_str(output_fh,handler_name, @sprintf  "\n# dims: %d   %d\n" dims[1] dims[2])

        return nothing
    end

    println("TODO: Add V0 and Vp0 to calculation_label!") 
    calculation_label = @sprintf "M%02d_N%02d_t%+5.3f_Vp%+5.3f_tsta%+5.3f_tend%+5.3f_tstep%+5.3f_Vsta%+5.3f_Vend%+5.3f_Vnum%04d" c[:L] c[:N] c[:t] c[:Vp] c[:time_min] c[:time_max] c[:time_step] c[:V_start] c[:V_end] c[:V_num]
    if c[:ee] == 2
        calculation_label = calculation_label*"_n02"
    end
    if c[:tdvp]
        calculation_label = calculation_label*"_tdvp"
    end
    if c[:boundary] == OBC
        calculation_label = calculation_label*"_obc"
    end
    if c[:first_order_trotter] 
        calculation_label = calculation_label*"_trotter1"
    end
    # Create output file handlers
    output_fh = FileOutputHandler(~c[:no_flush])
    
    # 2.1. output of particle entanglement (pe_01)
        Asize = c[:ee]
        handler_name = "particleEE"
        # function to convert data to string data = (t, entropies)
        out_str_pe_01 = data->@sprintf "%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E\n" data[1] data[2]...
        # open file
        path_pe_01 = joinpath(out_folder,@sprintf "particle_entanglement_n%02d_%s.dat" c[:ee] calculation_label)
        file_pe_01 = open(path_pe_01,"w")
        # add to file_handler
        OutputFileHandler.add!(output_fh,file_pe_01,out_str_pe_01,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), n=$(Asize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vnum=$(c[:V_num]), $(c[:boundary])\n")
        write_info(output_fh,handler_name,c)
        write_str(output_fh,handler_name,@sprintf "#%24s#%24s#%24s%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s\n" "t" "S₁(n=$(Asize))" "S₂(n=$(Asize))" "S₃(n=$(Asize))" "S₄(n=$(Asize))" "S₅(n=$(Asize))" "S₆(n=$(Asize))" "S₇(n=$(Asize))" "S₈(n=$(Asize))" "S₉(n=$(Asize))" "S₁₀(n=$(Asize))" "S₀₋₅(n=$(Asize))")
 
    # 2.2. output of spatial entanglement (se_02)
    ℓsize = Int(c[:L]/2)
    if c[:spatial]   
        handler_name = "spatialEE"
        # function to convert data to string data = (t, entropies)
        out_str_se_02 = (data)->@sprintf "%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E\n" data[1] data[2]...
        # open file
        path_se_02 = joinpath(out_folder,@sprintf "spatial_entanglement_l%02d_%s.dat" ℓsize calculation_label)
        file_se_02 = open(path_se_02,"w")
        # add to file_handler
        OutputFileHandler.add!(output_fh,file_se_02,out_str_se_02,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), l=$(ℓsize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vnum=$(c[:V_num]), $(c[:boundary])\n")
        write_info(output_fh,handler_name,c)
        write_str(output_fh,handler_name,@sprintf "#%24s#%24s#%24s%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s\n" "t" "S₁(n=$(ℓsize))" "S₂(n=$(ℓsize))" "S₃(n=$(ℓsize))" "S₄(n=$(ℓsize))" "S₅(n=$(ℓsize))" "S₆(n=$(ℓsize))" "S₇(n=$(ℓsize))" "S₈(n=$(ℓsize))" "S₉(n=$(ℓsize))" "S₁₀(n=$(ℓsize))" "S₀₋₅(n=$(ℓsize))")      
    end

    # 2.3. output of accessible entanglement (ae_03)
    if c[:spatial]   
        ℓsize = Int(c[:L]/2)
        handler_name = "accessibleEE"
        # function to convert data to string data = (t, entropies)
        out_str_ae_03 = (data)->@sprintf "%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E\n" data[1] data[2]...
        # open file
        path_ae_03 = joinpath(out_folder,@sprintf "accessible_entanglement_l%02d_%s.dat" ℓsize calculation_label)
        file_ae_03 = open(path_ae_03 ,"w")
        # add to file_handler
        OutputFileHandler.add!(output_fh,file_ae_03,out_str_ae_03,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), l=$(ℓsize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vnum=$(c[:V_num]), $(c[:boundary])\n")
        write_info(output_fh,handler_name,c)
        write_str(output_fh,handler_name,@sprintf "#%24s#%24s#%24s%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s\n" "t" "Sacc₁(ℓ=$(ℓsize))" "Sacc₂(ℓ=$(ℓsize))" "Sacc₃(ℓ=$(ℓsize))" "Sacc₄(ℓ=$(ℓsize))" "Sacc₅(ℓ=$(ℓsize))" "Sacc₆(ℓ=$(ℓsize))" "Sacc₇(ℓ=$(ℓsize))" "Sacc₈(ℓ=$(ℓsize))" "Sacc₉(ℓ=$(ℓsize))" "Sacc₁₀(ℓ=$(ℓsize))" "Sacc₀₋₅(ℓ=$(ℓsize))")      
    end

    # 2.4 debug printing
    if c[:debug]
        handler_name = "debug"
        # function to convert data to string data = (t, <psi|psi_inf>, <psi|psi_bot1> ...  <psi|psi_botn>)
        out_str_debug_04 = (data)->@sprintf "%24.12E%24.12E%s\n" data[1] data[2] join([@sprintf "%24.12E" sp for sp in data[3]], "")
        # open file
        path_debug_04 = joinpath(out_folder,@sprintf "debug_%s.dat" calculation_label)
        file_debug_04 = open(path_debug_04,"w")
        # add to file_handler
        OutputFileHandler.add!(output_fh,file_debug_04,out_str_debug_04,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), l=$(ℓsize), n=$(Asize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vnum=$(c[:V_num]), $(c[:boundary])\n")
        write_info(output_fh,handler_name,c)
        write_str(output_fh,handler_name,@sprintf "#%24s%24s%24s\n" "t" "<psi|psi_inf>" "<psi|psi_bot1> ..." )      
    end

    # 2.5 obdm printing
    if c[:obdm]
        handler_name = "obdm"
        # function to convert data to string data = (t, obdm entries)
        out_str_obdm_05 = (data)->@sprintf "%24.12E%s\n" data[1] join([@sprintf "%24.12E" sp... for sp in data[2]], "")
        # open file
        path_obdm_05 = joinpath(out_folder,@sprintf "obdm_%s.dat" calculation_label)
        file_obdm_05 = open(path_obdm_05,"w")
        # add to file_handler
        OutputFileHandler.add!(output_fh,file_obdm_05,out_str_obdm_05,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), l=$(ℓsize), n=$(Asize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vnum=$(c[:V_num]), $(c[:boundary])\n")
        write_info(output_fh,handler_name,c)
        write_str(output_fh,handler_name,@sprintf "#%24s%s\n" "t (|i-j|-->)" join([@sprintf "%24d" xi for xi in (-c[:N]+1):c[:N]], "") )      
    end 

    # 2.6 state snapshots
    # defined above: snapshot_label = @sprintf "M%02d_N%02d_t%+5.3f_Vp%+5.3f_tstep%+5.3f_Vsta%+5.3f_Vend%+5.3f_Vnum%04d" c[:L] c[:N] c[:t] c[:Vp] c[:time_step] c[:V_start] c[:V_end] c[:V_num]
    snapshot_sh = SnapshotHandler() 
    handler_name = "state"
    # function to convert data to string data = (t, obdm entries)
    function write_out(path,psi) 
        println("Save snapshot to: $(path)...")
        file = h5open(path,"w")
        write(file,"psi" ,psi)
        close(file)
        return nothing
    end
    function read_in(path) 
        println("Load snapshot from: $(path)...")
        file = h5open(path,"r")
        psi = read(file,"psi",MPS) 
        close(file)
        return psi
    end
    # open file
    # defined above: file_name_state 
    function path_generator_state_06(t::Float64) 
        return joinpath(states_out_folder,file_name_state(t)) 
    end
    # add to file_handler
    OutputFileHandler.add!(snapshot_sh,path_generator_state_06,write_out,read_in,c[:snapshots_every],handler_name) 
   

 # _____________3_Calculation______________________ 
    Random.seed!(c[:seed]) 
    tV_dmrg_ee_calclation_quench_gpu(c,output_fh,snapshot_sh) 

 # ________4_Output_Finalization___________________ 
    for (h_name,) in output_fh.handler_name_lookup
        write_str(output_fh,h_name,"\n\n Calculation finished at  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    end
    close(output_fh) 

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

