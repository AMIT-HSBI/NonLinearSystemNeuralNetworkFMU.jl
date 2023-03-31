using NonLinearSystemNeuralNetworkFMU

function genData(modelName::String, moFiles::Array{String}, N::Integer)
  workdir = datadir("sims", "$(modelName)_$(N)")
  options = OMOptions(workingDir=workdir,
                      clean=false,
                      commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  main(modelName, moFiles; options=options, reuseArtifacts=true, N=N)
end

function genAllData(modelName, N)
  moFiles = [srcdir("$modelName.mo")]
  (csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, N)
end
