using DrWatson
@quickactivate "SimpleLoop"

using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using DataFrames
using CSV

include(srcdir("isRight.jl"))

# Genrate data for SimpleLoop
modelName = "simpleLoop"
moFiles = [(srcdir("simpleLoop.mo"))]

# Model parameters
b = -0.5

"""
    runExample(n)

Run example for given number of data points n.
  1. Generate n data points
  2. Train ANN on filtered data set.
  3. Include ONNX into FMU
"""
function runExamples(n)
  workingDir = datadir("sims", modelName*"_$n")
  rm(workingDir, force=true, recursive=true)
  omOptions = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)
  dataGenOptions = NonLinearSystemNeuralNetworkFMU.DataGenOptions(method=RandomMethod(), n=n)

  (csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(
    modelName,
    moFiles;
    omOptions=omOptions,
    dataGenOptions=dataGenOptions,
    reuseArtifacts=false)
  mkpath(datadir("sims", "fmus"))
  cp(fmu, datadir("sims", "fmus", modelName*".fmu"), force=true)
  df =  CSV.read(csvFiles[1], DataFrame; ntasks=1)

  # Save (filtered) data
  name = basename(csvFiles[1])
  csvFile = datadir("exp_raw", split(name, ".")[1] * "_N$n." * split(name, ".")[2])
  mkpath(dirname(csvFile))
  CSV.write(csvFile, df)

  df_filtered = filter(row -> !isRight(row.s, row.r, row.y; b=b), df)
  csvFileFiltered = datadir("exp_pro", split(name, ".")[1] * "_N$n." * split(name, ".")[2])
  mkpath(dirname(csvFileFiltered))
  CSV.write(csvFileFiltered, df_filtered)

  # Train surrogate
  onnxFiles = String[]
  for prof in profilingInfo
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

  cp(fmu_onnx, datadir("sims", "fmus", modelName*".onnx_N$n.fmu"), force=true)
end

# Run examples
runExamples.([100, 500, 750, 1000])
