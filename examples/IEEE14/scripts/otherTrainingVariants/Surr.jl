using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using ChainRulesCore
using Zygote
#using BSON
using Flux
using LinearAlgebra
using FMI
using FMIImport

using Statistics
using Plots
using Metrics

#import DataFrames
#import CSV
#import InvertedIndices
#import StatsBase

using Random
import FMICore
using Libdl
using Surrogates

include("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/simpleLoopVariants/utils.jl")




function generateDataPoint(fmu, eqId, nInputs, row_vr, row, time)
    # Set input values and start values for output
    FMIImport.fmi2SetReal(fmu, row_vr, row)
    if time !== nothing
      FMIImport.fmi2SetTime(fmu, time)
    end
  
    # Evaluate equation
    # TODO: Supress stream prints, but Suppressor.jl is not thread safe
    status = fmiEvaluateEq(fmu, eqId)
    if status == fmi2OK
      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])
    end
  
    return status, row
end



comp, fmu, profilinginfo, vr, row_value_reference, eq_num, sys_num = prepare_fmu("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/temp-profiling/simpleLoop.c")

nInputs = 2
nOutputs = 1

num_samples = 100
lb = profilinginfo[1].boundary.min
ub = profilinginfo[1].boundary.max
fmi2OK = Cuint(0)


#Sampling
x = sample(num_samples,lb,ub,SobolSample())
function g(fmu, eq_num, nInputs, nOutputs, row_vr, inputs, time)
    row = Array{Float64}(undef, nInputs+nOutputs)
    row[1:nInputs] .= inputs
    _, in_out = generateDataPoint(fmu, eq_num, nInputs, row_vr, row, time)
    return in_out[nInputs+1:end]
end
# actual function to be optimized
row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(profilinginfo[1].usingVars,profilinginfo[1].iterationVariables))
f = x -> g(fmu, eq_num, nInputs, nOutputs, row_vr, x, nothing)[1]
y = f.(x)

surrogate = RadialBasis(x, y, lb, ub)

value = surrogate([0.0, 0.0])

#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,surrogate,RandomSample())