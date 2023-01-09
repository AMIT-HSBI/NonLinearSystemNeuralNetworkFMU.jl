# $ nohup time julia -e "include(\"IEEE_14_Buses.jl\");" &

using Revise
using BSON
using CSV
using DataFrames
using FMI
using NaiveONNX

import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU

cd(@__DIR__)

N = 100

modelName = "IEEE_14_Buses"
moFiles = ["IEEE_14_Buses.mo"]
workdir = joinpath(@__DIR__, modelName*"_$(N)")

main(modelName, moFiles; workdir=workdir, reuseArtifacts=true, clean=false, N=N)

dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])

# Train ONNX
onnxFiles = String[]
for (i, prof) in enumerate(profilingInfo)
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  push!(onnxFiles, onnxModel)
  nInputs = length(prof.usingVars)

  csvFile = joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv")
  @showtime trainONNX(csvFile, onnxModel, nInputs; nepochs=100, losstol=1e-8)
end

# Include ONNX into FMU
fmu_interface = joinpath(workdir, modelName*".interface.fmu")
tempDir = joinpath(workdir, "temp")
fmu_onnx = buildWithOnnx(fmu_interface,
                         modelName,
                         profilingInfo,
                         onnxFiles;
                         tempDir=workdir)
rm(tempDir, force=true, recursive=true)

# Simulate FMUs
#=
# FMI.jl is broken for this example. It needs forever to simulate, while OMSimulator is nice and fast
outputVars = ["B1.v", "B2.v", "B3.v", "B6.v", "B8.v"]

resultFile = joinpath(workdir, modelName*".csv")
fmu = fmiLoad(joinpath(workdir, modelName*".fmu"))
result = fmiSimulate(fmu, 0.0, 10.0; recordValues=outputVars, dtmax=0.0001)
fmiUnload(fmu)

df = DataFrame(time = result.values.t)
for i in 1:length(result.values.saveval[1])
  df[!, Symbol(outputVars[i])] = [val[i] for val in result.values.saveval]
end
CSV.write(resultFile, df)
=#
