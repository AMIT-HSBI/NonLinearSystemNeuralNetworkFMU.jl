using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU

modelName = "IEEE_14_Buses"
moFiles = [srcdir("IEEE_14_Buses.mo")]

function genData(modelName::String, moFiles::Array{String}, N::Integer)
  workdir = datadir("sims", "$(modelName)_$(N)")
  options = OMOptions(workingDir=workdir,
                      clean=false,
                      commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  main(modelName, moFiles; options=options, reuseArtifacts=true, N=N)
end

(csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, 1000)
