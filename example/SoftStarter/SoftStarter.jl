using Revise
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU
using NaiveONNX

modelName = "SoftStarter"
modelicaFile = joinpath(@__DIR__,"SoftStarter.mo")
# Generate data
(csvFiles, profilingInfo) = main("SoftStarter", modelicaFile, workdir=joinpath(@__DIR__,modelName), reuseArtifacts=true)

# Train ONNX
onnxFiles = String[]
for (i ,prof) in enumerate(profilingInfo)
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  push!(onnxFiles, onnxModel)
  nInputs = length(prof.usingVars)

  @showtime trainONNX(csvFiles[i], onnxModel, nInputs; nepochs=100, losstol=1e-4)
end
