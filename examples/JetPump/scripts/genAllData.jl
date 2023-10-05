using DrWatson
@quickactivate "JetPump"

using NonLinearSystemNeuralNetworkFMU
include(srcdir("flatten.jl"))

n = 100_000
modelName = "Scenario_02"
rootDir = "/mnt/home/aheuermann/workdir/phymos/JetPump"
moFiles = [joinpath(rootDir, "02_SourceModel", "Modelica", "JetPumpTool", "package.mo"), projectdir("models", "$(modelName).mo")]
workdir = datadir("sims", "$(modelName)_$(n)", "temp-flatmodel")

# Flatten Modelica model
#flatModel = flattentModelica(modelName, moFiles, datadir("sims", "$(modelName)_$(n)", "$(modelName)_flat.mo"), workdir=workdir)
#moFiles = [flatModel]
#flatModel = datadir("sims", "$(modelName)_$(n)", "$(modelName)_flat.mo")

#totalModel = saveTotalModel(modelName, moFiles, datadir("sims", "$(modelName)_$(n)", "$(modelName)_total.mo"), workdir=workdir)
#moFiles = [totalModel]

# Generate data
function genData(modelName::String, moFiles::Array{String}, n::Integer)
  workdir = datadir("sims", "$(modelName)_$(n)")
  omOptions = OMOptions(workingDir=workdir,
                        clean=false,
                        commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  dataGenOptions =  DataGenOptions(method=RandomMethod(), n=n, nBatches=Threads.nthreads())
  main(modelName, moFiles; omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=false)
end

(csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, n)
