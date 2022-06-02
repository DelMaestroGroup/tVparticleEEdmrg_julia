"""
Module FermionBasis contains the functions to generate a integer fermion basis,
translate to particle position basis, and locate elements in the basis
"""
module FermionBasis 

    export  
        get_int_fermion_basis,
        convert_basis_vector,
        get_position_int_basis

     
    include("fermion_basis_functions.jl") 
end