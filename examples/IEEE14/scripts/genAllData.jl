using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU

modelName = "IEEE_14_Buses"
moFiles = [srcdir("IEEE_14_Buses.mo")]

function genData(modelName::String, moFiles::Array{String}, n::Integer)
  workdir = datadir("sims", "$(modelName)_$(n)")
  options = OMOptions(workingDir=workdir,
                      clean=false,
                      commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  dataGenOptions = DataGenOptions(method=RandomMethod(), n=n, nBatches=Threads.nthreads())
  
  main(modelName, moFiles; omOptions=options, dataGenOptions=dataGenOptions, reuseArtifacts=true)
end

(csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, 1000)
