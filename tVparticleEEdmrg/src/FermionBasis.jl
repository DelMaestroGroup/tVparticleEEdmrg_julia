"""
Module FermionBasis contains the functions to generate a integer fermion basis,
translate to particle position basis, and locate elements in the basis.
It dynamically switches between Int64 and Int128 representations to allow for
fast implementation of lattices with <= 64 sites but also lattices up to 128 sites.
"""
module FermionBasis 

    export  
        get_int_fermion_basis,
        convert_basis_vector,
        get_position_int_basis

     
    include("fermion_basis_functions.jl") 
end