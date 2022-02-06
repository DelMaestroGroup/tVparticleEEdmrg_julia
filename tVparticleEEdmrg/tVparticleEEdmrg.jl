# Entanglement entropy of the t-V-V` Model at half-filling - dependence on interaction strength V

push!(LOAD_PATH, joinpath(dirname(@__FILE__), "src"))
 
using ArgParse 
using Printf 
using ProgressBars
using Dates
using Pkg
using DMRGEntanglementCalculation
using OutputFileHandler

Pkg.add("ITensors")
Pkg.add("ArgParse")
Pkg.add("ProgressBars")
Pkg.add("BenchmarkTools")


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
        "--spatial"
            help = "output the spatial entanglement entropy for ℓ = M/2"
            action = :store_true  
        "--no-flush"
            help = "do not flush write buffer to output files in after computation for each V" 
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
        "--V_start"
            metavar = "V_start"
            help = "start V"
            arg_type = Float64
            default = -2.0
        "--V_end"
            metavar = "V_end"
            help = "end V"
            arg_type = Float64
            default = 2.0
        "--V_step"
            metavar = "V_step"
            help = "step in V"
            arg_type = Float64
            default = 0.1 
        "--Vp"
            metavar = "Vp"
            help = "final Vp"
            arg_type = Float64
            default = 0.0
        "--t"
            metavar = "t"
            help = "t value"
            arg_type = Float64
            default = 1.0
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
    if c[:ee] != 1
        print("Currently the dmrg calculation only supports one particle entanglement entropy not n=",c[:ee], " . exit()")
        exit(1)
    end

 # _____________2_Output_Setup___________________
    if c[:out] === nothing
        out_folder = "./"
    else
        out_folder = c[:out]
    end   
    calculation_label = @sprintf "M%02d_N%02d_t%+5.3f_Vp%+5.3f_Vsta%+5.3f_Vend%+5.3f_Vstp%+5.3f" c[:L] c[:N] c[:t] c[:Vp] c[:V_start] c[:V_end] c[:V_step]
    # Create output file handlers
    output_fh = FileOutputHandler(~c[:no_flush])
    
    # 2.1. output of particle entanglement (pe_01)
        Asize = c[:ee]
        handler_name = "particleEE"
        # function to convert data to string data = (V, entropies)
        out_str_pe_01 = data->@sprintf "%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E\n" data[1] data[2]...
        # open file
        path_pe_01 = joinpath(out_folder,@sprintf "particle_entanglement_n%02d_%s.dat" c[:ee] calculation_label)
        file_pe_01 = open(path_pe_01,"w")
        # add to file_handler
        add!(output_fh,file_pe_01,out_str_pe_01,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), n=$(Asize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vstep=$(c[:V_step]), $(c[:boundary])\n")
        write_str(output_fh,handler_name, "# start time $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
        write_str(output_fh,handler_name,@sprintf "#%24s#%24s#%24s%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s\n" "V" "S₁(n=$(Asize))" "S₂(n=$(Asize))" "S₃(n=$(Asize))" "S₄(n=$(Asize))" "S₅(n=$(Asize))" "S₆(n=$(Asize))" "S₇(n=$(Asize))" "S₈(n=$(Asize))" "S₉(n=$(Asize))" "S₁₀(n=$(Asize))" "S₀₋₅(n=$(Asize))")
 
    # 2.2. output of spatial entanglement (se_02)
    if c[:spatial]   
        ℓsize = Int(c[:L]/2)
        handler_name = "spatialEE"
        # function to convert data to string data = (V, entropies)
        out_str_se_02 = (data)->@sprintf "%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E%24.12E\n" data[1] data[2]...
        # open file
        path_se_02 = joinpath(out_folder,@sprintf "spatial_entanglement_l%02d_%s.dat" ℓsize calculation_label)
        file_se_02 = open(path_se_02,"w")
        # add to file_handler
        add!(output_fh,file_se_02,out_str_se_02,handler_name)
        # write initial header
        write_str(output_fh,handler_name, "# M=$(c[:L]), N=$(c[:N]), Vp=$(c[:Vp]), t=$(c[:t]), l=$(ℓsize), Vstart=$(c[:V_start]), Vstop=$(c[:V_end]), Vstep=$(c[:V_step]), $(c[:boundary])\n")
        write_str(output_fh,handler_name, "# start time $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
        write_str(output_fh,handler_name,@sprintf "#%24s#%24s#%24s%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s#%24s\n" "V" "S₁(n=$(ℓsize))" "S₂(n=$(ℓsize))" "S₃(n=$(ℓsize))" "S₄(n=$(ℓsize))" "S₅(n=$(ℓsize))" "S₆(n=$(ℓsize))" "S₇(n=$(ℓsize))" "S₈(n=$(ℓsize))" "S₉(n=$(ℓsize))" "S₁₀(n=$(ℓsize))" "S₀₋₅(n=$(ℓsize))")      
    end

 # _____________3_Calculation______________________ 
    tV_dmrg_ee_calclation_equilibrium(c,output_fh)

 # ________4_Output_Finalization___________________ 
    for (h_name,) in output_fh.handler_name_lookup
        write_str(output_fh,h_name,"\n\n Calculation finished at  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    end
    close(output_fh) 

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
