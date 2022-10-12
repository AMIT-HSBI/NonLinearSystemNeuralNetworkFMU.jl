using Revise
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NonLinearSystemNeuralNetworkFMU

modelName = "Furnace"
moFiles = ["/mnt/home/aheuermann/workdir/Testitesttest/testlibs/clara_om/ClaRa/package.mo", joinpath(@__DIR__,"$modelName.mo")]
workdir = joinpath(@__DIR__, modelName)

# Generate data
(csvFiles, fmu, profilingInfo) = main(modelName, moFiles, workdir=workdir, reuseArtifacts=false)
