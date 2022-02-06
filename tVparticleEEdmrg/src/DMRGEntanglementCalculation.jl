"""
Module DMRGEntanglementCalculation contains all function to calcuate the entangleent entropy
    and write it to output files.

Usage: 
     
"""
module DMRGEntanglementCalculation
    using Base
    using OutputFileHandler

    export  
        tV_dmrg_ee_calclation_equilibrium,
        BdryCond, PBC, OBC
     
    """
    Boundary conditions.
    """
    @enum BdryCond PBC OBC
    @doc "Periodic boundary conditions." PBC
    @doc "Open boundary conditions." OBC

    include("dmrg_entanglement_calculation_functions.jl")
end