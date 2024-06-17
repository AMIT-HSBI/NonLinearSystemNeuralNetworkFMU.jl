using NonLinearSystemNeuralNetworkFMU

function genData(modelName::String, moFiles::Array{String}, n::Integer)
  workdir = datadir("sims", "$(modelName)_$(n)")
  omOptions = OMOptions(workingDir=workdir,
                        clean=false,
                        commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  dataGenOptions = NonLinearSystemNeuralNetworkFMU.DataGenOptions(method=RandomMethod(), n=n)

  main(modelName, moFiles; omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=true)
end

function genAllData(modelName, n)
  moFiles = [srcdir("$modelName.mo")]
  (csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, n)
end
