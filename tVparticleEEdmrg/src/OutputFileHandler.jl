"""
Module OutputFileHandler defines the type FileOutputHandler which collects lists of
IOStream files, functions to convert data to strings to write to these files, 
and handler names. 
In addition, implements snapshot file handler to save states based on t.

Usage: 
    using OutputFileHandler
    using Printf
    # create empty handler  
    fh = FileOutputHandler()
    # add file and output
    add!(fh,open("test.txt","w"),x-> (@sprintf "a=%d,b=%d,c=%d\n" x[1] x[2] x[3]),"results")
    # write to the file corresponding to "results"
    write(fh,"results",[1,2,3])
    write(fh,"results",[1,4,5])
    write(fh,"results",[-1,-2,-3])
    # close all files
    close(fh)
"""
module OutputFileHandler
    using Base

    export  
        FileOutputHandler,
        SnapshotHandler,
        add!,
        write,
        write_str, 
        close,
        get_path,
        read,
        time_for_snapshot
     

    include("output_file_handler_functions.jl")
    include("snapshot_handler_functions.jl")
end