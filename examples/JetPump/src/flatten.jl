using OMJulia
using NonLinearSystemNeuralNetworkFMU: OpenModelicaError

function flattentModelica(modelName::String, moFiles::Array{String}, flatModel::String; workdir::String)

  mkpath(workdir)
  omc = OMJulia.OMCSession()

  for file in moFiles
    msg = sendExpression(omc, "loadFile(\"$(file)\")")
    if (msg != true)
      msg = sendExpression(omc, "getErrorString()")
      write(logFile, msg*"\n")
      throw(OpenModelicaError("Failed to load file $(file)!", abspath(logFilePath)))
    end
  end

  sendExpression(omc, "setCommandLineOptions(\"--flatModelica --modelicaOutput\")")
  sendExpression(omc, "cd(\"$(workdir)\")")
  flatModelicaCode = sendExpression(omc, "instantiateModel($(modelName))")

  OMJulia.sendExpression(omc, "quit()",parsed=false)

  # Remove ' in model name
  flatModelicaCode = replace(flatModelicaCode, "'$(modelName)'" => modelName)

  # TODO: Add annotation(experiment(StopTime = 100.0, Interval = 0.1));
  # This is removed from the flattened model

  open(flatModel,"w") do file
    write(file, flatModelicaCode)
  end

  return flatModel
end
