"""
We use the code from https://github.com/orialb/TimeEvoMPS.jl (Ori Alberton) 
and adjust it to our needs.
The idea is based on 
    [1] Vidal, G. (2004). Efficient Simulation of One-Dimensional Quantum Many-Body Systems.
        Physical Review Letters, 93(4), 040502. https://doi.org/10.1103/PhysRevLett.93.040502
    [2] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
        Unifying time evolution and optimization with matrix product states. Physical Review B, 94(16).
        https://doi.org/10.1103/PhysRevB.94.165116
"""

module Tdvp 

using ITensors
using KrylovKit

    export  
        tdvp_step! 
     

    include("tdvp_functions.jl")
end