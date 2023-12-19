using NonLinearSystemNeuralNetworkFMU
using BSON
using FMI
using FMIImport
import FMICore
using Libdl


function fmiEvaluateJacobian(comp::FMICore.FMU2Component, eq::Integer, vr::Array{FMI.fmi2ValueReference}, x::Array{Float64})::Tuple{FMI.fmi2Status, Array{Float64}}
    # wahrscheinlich braucht es eine c-Funktion, die nicht nur einen pointer auf die Jacobi-Matrix returned, sondern gleich die Auswertung an der Stelle x
    # diese Funktion muss auch ein Argument res nehmen welches dann die Evaluation enthält.?
  
    @assert eq>=0 "Residual index has to be non-negative!"
  
    FMIImport.fmi2SetReal(comp, vr, x)
  
    # this is a pointer to Jacobian matrix in row-major-format or NULL in error case.
    fmiEvaluateJacobian = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateJacobian)
  
    jac = Array{Float64}(undef, length(x)*length(x))
  
    #eqCtype = Csize_t(eq)
  
    status = ccall(fmiEvaluateJacobian,
                   Cuint,
                   (Ptr{Nothing}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
                   comp.compAddr, eq, x, jac)
  
    return status, jac
end

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


#IEEE14 "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-extendfmu/IEEE_14_Buses.fmu"
#       "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/profilingInfo.bson"

#simpleLoop /home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/unzipped_fmu_sl/simpleLoop.fmu
#           /home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/profilingInfo.bson

fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-extendfmu/IEEE_14_Buses.fmu")
comp = FMI.fmiInstantiate!(fmu)
FMI.fmiSetupExperiment(comp)
FMI.fmiEnterInitializationMode(comp)
FMI.fmiExitInitializationMode(comp)

profilinginfo = getProfilingInfo("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/profilingInfo.bson")

vr = FMI.fmiStringToValueReference(fmu, profilinginfo[1].iterationVariables)


eq_num = profilinginfo[1].eqInfo.id
modelfile_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-profiling/IEEE_14_Buses.c"

sys_num = parse_modelfile(modelfile_path, eq_num)

x = rand(length(profilinginfo[1].iterationVariables),)
status, jac = fmiEvaluateJacobian(comp, sys_num, vr, x)

#IEEE14 [1:5,1:5]
num_it_vars = length(profilinginfo[1].iterationVariables)
display(reshape(jac, (num_it_vars, num_it_vars)))


# when eqNumber == 0, then 1x1 output is okay, NxN output only [1,1] is okay other values are e-100 or so
# when eqNumber == eq_num, then crash