using Revise
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU
using BSON: @load

modelName = "Furnace"
moFiles = ["/mnt/home/aheuermann/workdir/Testitesttest/testlibs/clara_om/ClaRa/package.mo", joinpath(@__DIR__,"$modelName.mo")]
workdir = joinpath(@__DIR__, modelName)

# Generate data
(csvFiles, fmu, profilingInfo) = main(modelName, moFiles, workdir=workdir, reuseArtifacts=false)

# Run Python code to train ANN
run(Cmd(`python3 trainTensorFlow.py`, dir = @__DIR__))

# generate ONNX FMUs
interfaceFmu = joinpath(@__DIR__, modelName, modelName*".interface.fmu")
@load joinpath(@__DIR__, modelName, "profilingInfo.bson") profilingInfo
profilingInfo = Array{ProfilingInfo}(profilingInfo)
onnxDir = abspath(@__DIR__, modelName, "onnx")
onnxFiles = joinpath.(onnxDir, ["eq_1933.onnx", "eq_2134.onnx"])
pathToFmu = buildWithOnnx(interfaceFmu, modelName, profilingInfo, onnxFiles; tempDir=workdir)
rm(joinpath(workdir, "FMU"), recursive=true, force=true)
