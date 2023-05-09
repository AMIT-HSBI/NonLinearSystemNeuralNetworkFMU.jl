using DrWatson
@quickactivate "ScalableTranslationStatistics"

using NonLinearSystemNeuralNetworkFMU
using CairoMakie
using CSV
using DataFrames

for size in [5, 10, 20, 40, 80]
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
  shortName = split(modelName, ".")[end]

  profilingInfos = NonLinearSystemNeuralNetworkFMU.getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))

  prof = profilingInfos[1]

  simDir = datadir("sims", shortName)

  csvFileData = joinpath(simDir, "data", "eq_$(prof.eqInfo.id).csv")
  df_data = CSV.read(csvFileData, DataFrame; ntasks=1)

  refCsvFile = joinpath(simDir, "temp-profiling", modelName*"_res.csv")
  df_ref = CSV.read(refCsvFile, DataFrame; ntasks=1)

  resultCsvFile = datadir("exp_raw", shortName, "$(shortName)_res.onnx.csv")
  df_surr = CSV.read(resultCsvFile, DataFrame; ntasks=1)

  fig = plotTrainArea(prof.iterationVariables, df_ref, df_surrogate = df_surr, df_trainData = df_data)
  mkpath(dirname(plotsdir(modelName, "$(shortName)_trainData.svg")))
  save(plotsdir(modelName, "$(shortName)_trainData.svg"), fig)
end
