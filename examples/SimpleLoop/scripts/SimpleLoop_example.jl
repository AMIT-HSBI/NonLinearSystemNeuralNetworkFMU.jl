using DrWatson
@quickactivate "SimpleLoop"

using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using Revise
using DataFrames
using CSV
using CairoMakie

# Genrate data for SimpleLoop
modelName = "simpleLoop"
moFiles = [(srcdir("simpleLoop.mo"))]

# Model parameters
b = -0.5

"""
Filter data for `x = r*s + b -y <= y` to get uniqe data points.
"""
function isRight(s,r,y; b)
  x = r*s + b -y
  return x > y
end

"""
    runExample(N)

Run example for given number of data points N.
  1. Generate N data points
  2. Train ANN on filtered data set.
  3. Include ONNX into FMU
"""
function runExamples(N)
  workingDir = datadir("sims", modelName*"_$N")
  rm(workingDir, force=true, recursive=true)
  options = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)

  (csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; options=options, reuseArtifacts=false, N=N)
  df =  CSV.read(csvFiles[1], DataFrame; ntasks=1)

  # Save (filtered) data
  name = basename(csvFiles[1])
  csvFile = datadir("exp_raw", split(name, ".")[1] * "_N$N." * split(name, ".")[2])
  if !isdir(dirname(csvFile))
    mkpath(dirname(csvFile))
  end
  CSV.write(csvFile, df)

  df_filtered = filter(row -> !isRight(row.s, row.r, row.y; b=b), df)
  csvFileFiltered = datadir("exp_pro", split(name, ".")[1] * "_N$N." * split(name, ".")[2])
  if !isdir(dirname(csvFileFiltered))
    mkpath(dirname(csvFileFiltered))
  end
  CSV.write(csvFileFiltered, df_filtered)

  # Train surrogate
  onnxFiles = String[]
  for (i, prof) in enumerate(profilingInfo)
    onnxModel = abspath(joinpath(workingDir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)

    @showtime trainONNX(csvFileFiltered, onnxModel, prof.usingVars, prof.iterationVariables; nepochs=1000, losstol=1e-8)
  end

  # Include ONNX into FMU
  fmu_interface = joinpath(workingDir, modelName*".interface.fmu")
  tempDir = joinpath(workingDir, "temp")
  fmu_onnx = buildWithOnnx(fmu_interface,
                          modelName,
                          profilingInfo,
                          onnxFiles;
                          tempDir=tempDir)

  if !isdir(datadir("sims", "fmus"))
    mkpath(datadir("sims", "fmus"))
  end
  cp(fmu_onnx, datadir("sims", "fmus", modelName*".onnx_N$N.fmu"), force=true)
end

# Run examples
runExamples.([100, 500, 750, 1000])
