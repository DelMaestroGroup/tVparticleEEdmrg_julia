"""
Module DMRGEntanglementCalculation contains function to calcuate the engangle entropy 
    and write it to output files.
    This imports GPU versions of functions. This module requires ITensorGPU.

Usage: 
     
"""
module DMRGEntanglementCalculationGPU
    using Base
    using OutputFileHandler
    using ITensors
    using ITensorGPU
    using Utils

    export  
        tV_dmrg_ee_calclation_equilibrium,
        tV_dmrg_ee_calclation_quench,
        tV_dmrg_ee_calclation_quench_gpu,
        BdryCond, PBC, OBC
     
    """
    Boundary conditions.
    """
    @enum BdryCond PBC OBC
    @doc "Periodic boundary conditions." PBC
    @doc "Open boundary conditions." OBC

    include("dmrg_entanglement_calculation_functions.jl")
    include("dmrg_entanglement_quench_calculation_functions.jl")
    include("dmrg_entanglement_quench_calculation_functions_gpu.jl")
end