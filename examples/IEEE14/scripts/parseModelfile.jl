"""
equation index: 16
indexNonlinear: 1
"""


# read file contents, one line at a time
eq_num = 1403
file_path = "examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-profiling/IEEE_14_Buses.c"

function parse_modelfile(modelfile_path, eq_num)
    conc_string = "equation index: " * string(eq_num)
    nonlin_string = "indexNonlinear: "
    regex_expression = r"(?<=indexNonlinear: )(\d+)"
    open(modelfile_path) do f
        # line_number
        found = false
        # read till end of file
        while !eof(f) 
            # read a new / next line for every iteration		 
            s = readline(f)

            if found && occursin(nonlin_string, s)
                line_matches = match(regex_expression, s)
                return parse(Int64, line_matches.match)
            end

            if occursin(conc_string, s)
                # read the next line
                found = true
            end
        end
        println("not found")
        return nothing
    end
end


sys_num = parse_modelfile(file_path, eq_num) #4