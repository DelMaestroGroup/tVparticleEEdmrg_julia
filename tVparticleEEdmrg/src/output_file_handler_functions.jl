using Base

"""Data type to handle file outputs"""
mutable struct FileOutputHandler
    files_list::Vector{IOStream}
    data_to_file_functions::Vector{Function}
    handler_name_lookup::Dict{String,Int64}
    nEntries::Int64
    flush_on_write::Bool
    open::Bool
    blocked_for_write_str::Vector{Bool}
end
"""Initialize an empty file output handler"""
FileOutputHandler(flush_on_write::Bool) = FileOutputHandler(Vector{IOStream}(), Vector{Function}(), Dict{String,Int64}(), 0, flush_on_write, true, Vector{Bool}())
FileOutputHandler() = FileOutputHandler(Vector{IOStream}(), Vector{Function}(), Dict{String,Int64}(), 0, true, true, Vector{Bool}())
function add!(fh::FileOutputHandler, file::IOStream, data_to_file_function::Function, handler_name::String; blocked_for_write::Bool=false)
    if ~fh.open
        error("Try to add to OutputFileHandler that is already closed.")
    end
    if haskey(fh.handler_name_lookup, handler_name)
        error("File handler names must be unique. Tried to add new file handler with already existing name ", handler_name, ".")
    end
    fh.nEntries += 1
    push!(fh.files_list, file)
    push!(fh.data_to_file_functions, data_to_file_function)
    push!(fh.blocked_for_write_str, blocked_for_write)
    fh.handler_name_lookup[handler_name] = fh.nEntries
    return nothing
end
"""Get file from handler name via handler_name_lookup dict"""
function _get_file(fh::FileOutputHandler, handler_name::String)
    index = fh.handler_name_lookup[handler_name]
    return fh.files_list[index]
end
"""Get blocked for write status from handler name via handler_name_lookup dict"""
function _get_blocked(fh::FileOutputHandler, handler_name::String)
    index = fh.handler_name_lookup[handler_name]
    return fh.blocked_for_write_str[index]
end
"""Write data to file. File is chosen by handler_name, data to string function is the correponding 
function in data_to_file_functions."""
function Base.write(fh::FileOutputHandler, handler_name::String, data...)
    if ~fh.open
        error("Try to write to OutputFileHandler that is already closed.")
    end
    index = fh.handler_name_lookup[handler_name]
    file = fh.files_list[index]
    write_str = fh.data_to_file_functions[index](data)
    write_flush(file, write_str, fh.flush_on_write)
    return nothing
end
"""Just run the data_to_file_functions corresponding to handler_name. This allows for more complicated write to file logic."""
function run(fh::FileOutputHandler, handler_name::String, data...)
    index = fh.handler_name_lookup[handler_name]
    file = fh.files_list[index]
    fh.data_to_file_functions[index](file, data)
    return nothing
end
"""Write string to file. File is chosen by handler_name."""
function write_str(fh::FileOutputHandler, handler_name::String, str::String)
    if ~fh.open
        error("Try to write to OutputFileHandler that is already closed.")
    end
    if _get_blocked(fh, handler_name)
        error("The file with handler name $(handler_name) is blocked for writing strings to it.")
    end
    file = _get_file(fh, handler_name)
    write_flush(file, str, fh.flush_on_write)
    return nothing
end
"""Write string to all files in handler."""
function write_str(fh::FileOutputHandler, str::String)
    if ~fh.open
        error("Try to write to OutputFileHandler that is already closed.")
    end
    for (file, blocked) in zip(fh.files_list, fh.blocked_for_write_str)
        if ~(blocked)
            write_flush(file, str, fh.flush_on_write)
        end
    end
    return nothing
end
"""Close all files."""
function Base.close(fh::FileOutputHandler)
    if ~fh.open
        error("Try to close OutputFileHandler that is already closed.")
    end
    for file in fh.files_list
        close(file)
    end
    fh.open = false
    return nothing
end

"""Use 'write' to write string to IOstream (e.g. write to a file) and flush IOstream if toflush is true."""
function write_flush(stream::IO, str::String, toflush::Bool=true)
    write(stream, str)
    if toflush
        flush(stream)
    end
    return nothing
end