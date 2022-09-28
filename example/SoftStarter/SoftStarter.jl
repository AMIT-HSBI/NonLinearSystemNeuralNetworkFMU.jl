using Revise
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using FMI

modelName = "SoftStarter"
moFiles = [joinpath(@__DIR__,"SoftStarter.mo")]
workdir = joinpath(@__DIR__, modelName)

# Generate data
(csvFiles, _, profilingInfo) = main("SoftStarter", moFiles, workdir=workdir, reuseArtifacts=true)

# Train ONNX
onnxFiles = String[]
for (i ,prof) in enumerate(profilingInfo)
  onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
  push!(onnxFiles, onnxModel)
  nInputs = length(prof.usingVars)

  @showtime trainONNX(csvFiles[i], onnxModel, nInputs; nepochs=100, losstol=1e-4)
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
outputVars = ["loadInertia.flange_a.tau", "loadInertia.flange_b.tau"]

resultFile = joinpath(workdir, modelName*".csv")
fmu = fmiLoad(joinpath(workdir, modelName*".fmu"))
result = fmiSimulate(fmu, 0.0, 1.0; recordValues=outputVars, dtmax=0.0001)
df = DataFrame(result, outputVars)
fmiUnload(fmu)

CSV.write(resultFile, df)
