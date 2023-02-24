using NaiveONNX
using NonLinearSystemNeuralNetworkFMU

function runScalableTranslationStatistics(lib::String, modelName::String; size::Int, N::Int=100)
  # Get lib and model
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  workdir = datadir("sims", split(modelName, ".")[end])

  # Generate training data
  options = OMOptions(workingDir=workdir)
  (csvFiles, fmu, profilingInfo) = main(modelName, [lib], options=options, reuseArtifacts=false, N=N)

  # Save data artifacts
  csvDir = datadir("exp_raw", split(modelName, ".")[end])
  if !isdir(csvDir)
    mkpath(csvDir)
  end
  for file in csvFiles
    cp(file, joinpath(csvDir, basename(file)), force=true)
  end

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
  cp(fmu_onnx, joinpath(fmuDir, modelName*".fmu"), force=true)

  return (profilingInfo, fmu, fmu_onnx)
end
