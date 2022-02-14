"""
Module DMRGEntanglementCalculation contains all function to calcuate the entangleent entropy
    and write it to output files.

Usage: 
     
"""
module DMRGEntanglementCalculation
    using Base
    using OutputFileHandler
    using Utils

    export  
        tV_dmrg_ee_calclation_equilibrium,
        tV_dmrg_ee_calclation_quench,
        BdryCond, PBC, OBC
     
    """
    Boundary conditions.
    """
    @enum BdryCond PBC OBC
    @doc "Periodic boundary conditions." PBC
    @doc "Open boundary conditions." OBC

    include("dmrg_entanglement_calculation_functions.jl")
    include("dmrg_entanglement_quench_calculation_functions.jl")
end