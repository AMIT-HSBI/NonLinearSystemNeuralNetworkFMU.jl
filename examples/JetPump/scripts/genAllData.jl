using DrWatson
@quickactivate "JetPump"

using NonLinearSystemNeuralNetworkFMU

modelName = "JetPumpInverse"
rootDir = "/mnt/home/aheuermann/workdir/phymos/JetPump"
moFiles = [joinpath(rootDir, "02_SourceModel", "Modelica", "JetPumpTool", "package.mo"), srcdir("$(modelName).mo")]

function genData(modelName::String, moFiles::Array{String}, n::Integer)
  workdir = datadir("sims", "$(modelName)_$(n)")
  omOptions = OMOptions(workingDir=workdir,
                        clean=false,
                        commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  dataGenOptions =  DataGenOptions(method=RandomMethod(), n=n, nBatches=Threads.nthreads())
  main(modelName, moFiles; omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=false)
end

(csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, 100_000)
