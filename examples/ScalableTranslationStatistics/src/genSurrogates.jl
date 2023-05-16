using NaiveONNX
using NonLinearSystemNeuralNetworkFMU

function genSurrogate(lib::String, modelName::String; N::Int=100)
  # Get lib and model
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  workdir = datadir("sims", split(modelName, ".")[end])

  # Generate training data
  omOptions = OMOptions(workingDir=workdir)
  dataGenOptions = DataGenOptions(method=RandomWalkMethod(delta=0.01),
                                  n = 1000,
                                  nBatches = 100)
  (csvFiles, fmu, profilingInfo) = main(modelName, [lib], omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=false)

  # Train ONNX
  onnxFiles = String[]
  for (i ,prof) in enumerate(profilingInfo)
    onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)

    @showtime trainONNX(csvFiles[i], onnxModel, prof.usingVars, prof.iterationVariables; nepochs=100, losstol=1e-4)
  end

  # Include ONNX into FMU
  fmu_interface = joinpath(workdir, modelName*".interface.fmu")
  tempDir = joinpath(workdir, "temp-onnx")
  @showtime fmu_onnx = buildWithOnnx(fmu_interface,
                                     modelName,
                                     profilingInfo,
                                     onnxFiles;
                                     tempDir=tempDir)

  # Save FMU artifacts
  fmuDir = datadir("sims", split(modelName, ".")[end], "fmus")
  if !isdir(fmuDir)
    mkpath(fmuDir)
  end
  cp(fmu, joinpath(fmuDir, modelName*".fmu"), force=true)
  cp(fmu_onnx, joinpath(fmuDir, modelName*".onnx.fmu"), force=true)

  return (profilingInfo, fmu, fmu_onnx)
end
