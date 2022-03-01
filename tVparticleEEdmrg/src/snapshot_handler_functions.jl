using Base

"""Data type to handle snapshots"""
mutable struct SnapshotHandler
    path_generator_list::Vector{Function}
    data_to_file_functions::Vector{Function}
    data_from_file_functions::Vector{Function}
    snapshot_frequency_list::Vector{Int64}
    handler_name_lookup::Dict{String, Int64}
    nEntries::Int64
    next_snapshot_list::Vector{Int64}
end
"""Initialize an empty snapshot handler"""
SnapshotHandler() = SnapshotHandler(Vector{Function}(),Vector{Function}(),Vector{Function}(),Vector{Int64}(),Dict{String,Int64}(),0,Vector{Int64}())
function add!(sh::SnapshotHandler,path_template::Function,data_to_file_function::Function,data_from_file_function::Function,updata_every_steps::Int64,handler_name::String) 
    if haskey(sh.handler_name_lookup,handler_name)
        error("Snashot handler names must be unique. Tried to add new snapshot handler with already existing name ", handler_name,".")
    end
    sh.nEntries += 1
    push!(sh.path_generator_list,path_template)
    push!(sh.data_to_file_functions,data_to_file_function) 
    push!(sh.snapshot_frequency_list,updata_every_steps)
    push!(sh.data_from_file_functions,data_from_file_function) 
    push!(sh.next_snapshot_list,updata_every_steps)
    sh.handler_name_lookup[handler_name] = sh.nEntries
    return nothing
end
"""Get path from handler name via handler_name_lookup dict"""
function get_path(sh::SnapshotHandler,handler_name::String,replace_params)
    index = sh.handler_name_lookup[handler_name] 
    return sh.path_generator_list[index](replace_params...)
end 
"""Write data to file. File is chosen by handler_name, data_to_file_functions function is the correponding 
function in data_to_file_functions that is called."""
function Base.write(sh::SnapshotHandler,handler_name::String,format_replace_params,data...)  
    path = get_path(sh,handler_name,format_replace_params...)
    index = sh.handler_name_lookup[handler_name]
    sh.data_to_file_functions[index](path,data...)
    return nothing
end
"""Read data from file. File is chosen by handler_name, data_to_file_functions function is the correponding 
function in data_to_file_functions that is called."""
function Base.read(sh::SnapshotHandler,handler_name::String,format_replace_params) 
    path = get_path(sh,handler_name,format_replace_params)
    index = sh.handler_name_lookup[handler_name] 
    return sh.data_from_file_functions[index](path)
end    
"""Checks if it is time to do a snapshot."""
function time_for_snapshot(sh::SnapshotHandler,handler_name::String,i::Int64)
    index = sh.handler_name_lookup[handler_name]
    if sh.snapshot_frequency_list[index] == 0
        return false
    end
    # need to check like this as we can only save after moving to cpu before a measurement 
    # and steps in between measurements are variable
    do_snapshot =  i >= sh.next_snapshot_list[index]
    if do_snapshot
        sh.next_snapshot_list[index] += sh.snapshot_frequency_list[index]
    end
    return do_snapshot
end