using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using BSON

N = 1000
modelName = "IEEE_14_Buses"

workdir = datadir("sims", "$(modelName)_$(N)")
dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])

# Train ONNX
onnxFiles = String[]
for (i, prof) in enumerate(profilingInfo)
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  push!(onnxFiles, onnxModel)

  csvFile = joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv")
  @showtime trainONNX(csvFile, onnxModel, prof.usingVars, prof.iterationVariables; nepochs=100, losstol=1e-8)
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
