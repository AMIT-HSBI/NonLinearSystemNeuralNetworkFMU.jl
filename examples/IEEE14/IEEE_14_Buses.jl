# $ nohup time julia --threads auto -e "include(\"IEEE_14_Buses.jl\");" &
# Run julia with --threads auto

#using Revise
using BSON
using CSV
using DataFrames
using FMI
using NaiveONNX

import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU

cd(@__DIR__)

N = 20000

modelName = "IEEE_14_Buses"
moFiles = ["IEEE_14_Buses.mo"]
workdir = joinpath(@__DIR__, modelName*"_$(N)")

main(modelName, moFiles; workdir=workdir, reuseArtifacts=false, clean=false, N=N)

dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])
fmu_interface = joinpath(workdir, modelName*".interface.fmu")

# Generate Data
@info "Generate training data"
csvFiles = String[]
for prof in profilingInfo
  eqIndex = prof.eqInfo.id
  inputVars = prof.usingVars
  outputVars = prof.iterationVariables
  minBoundary = prof.boundary.min
  maxBoundary = prof.boundary.max

  fileName = abspath(joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv"))
  csvFile = generateTrainingData(fmu_interface, fileName, eqIndex, inputVars, minBoundary, maxBoundary, outputVars; N = 4000)
  push!(csvFiles, csvFile)
end

# Train ONNX
onnxFiles = String[]
for (i, prof) in enumerate(profilingInfo)
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  push!(onnxFiles, onnxModel)
  nInputs = length(prof.usingVars)
  csvFile = joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv")
  epochs = i==1 ? 10000 : 100
  @showtime trainONNX(csvFile, onnxModel, nInputs; nepochs=epochs, losstol=1e-8)
end

# Include ONNX into FMU
tempDir = joinpath(workdir, "temp-onnx")
fmu_onnx = buildWithOnnx(fmu_interface,
                         modelName,
                         profilingInfo,
                         onnxFiles;
                         tempDir=workdir)
rm(tempDir, force=true, recursive=true)

#=
# Simulate FMUs first time
resultFile = "model_res_01.csv"
cmd = `OMSimulator IEEE_14_Buses.onnx.fmu --stopTime=10 --resultFile=$(resultFile)`
out = IOBuffer()
err = IOBuffer()
try
  @info "Running OMSimulator first time"
  @showtime run(pipeline(Cmd(cmd, dir=workdir); stdout=out, stderr=err))
  println(String(take!(out)))
catch e
  println(String(take!(err)))
  rethrow(e)
finally
  close(out)
  close(err)
end

allUsedVars = unique(vcat([prof.usingVars for prof in profilingInfo]...))
(allMin, allMax) = NonLinearSystemNeuralNetworkFMU.minMaxValues(joinpath(workdir, resultFile), allUsedVars; epsilon=0.05)
for prof in profilingInfo
  idx = findall(elem -> elem in prof.usingVars, allUsedVars)

  @info "Old min,max"
  @show prof.boundary.min, prof.boundary.max
  prof.boundary.min .= allMin[idx]
  prof.boundary.max .= allMax[idx]
  @info "New min,max"
  @show prof.boundary.min, prof.boundary.max
end

# Generate more trainng data
for prof in profilingInfo[1:1]
  eqIndex = prof.eqInfo.id
  inputVars = prof.usingVars
  outputVars = prof.iterationVariables
  minBoundary = prof.boundary.min
  maxBoundary = prof.boundary.max

  fileName = abspath(joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv"))
  generateTrainingData(fmu_interface, fileName, eqIndex, inputVars, minBoundary, maxBoundary, outputVars; N = 6000, append = true)
end

for (i, prof) in enumerate(profilingInfo[1:1])
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  nInputs = length(prof.usingVars)
  csvFile = joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv")
  @showtime trainONNX(csvFile, onnxModel, nInputs; nepochs=2000, losstol=1e-8)
end

tempDir = joinpath(workdir, "temp")
fmu_onnx = buildWithOnnx(fmu_interface,
                         modelName,
                         profilingInfo,
                         onnxFiles;
                         tempDir=workdir)
rm(tempDir, force=true, recursive=true)
=#
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
