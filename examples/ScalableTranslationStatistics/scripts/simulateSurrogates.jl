using DrWatson
@quickactivate "ScalableTranslationStatistics"

using BSON: @save

include(srcdir("simulateFMU.jl"))

sizes = [5, 10, 20, 40, 80]

outputVars = ["outputs[1]", "outputs[2]", "outputs[3]", "outputs[4]", "outputs[5]", "outputs[6]", "outputs[7]", "outputs[8]"]

for size in sizes
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
  shortName = split(modelName, ".")[end]

  benchmarkFile = datadir("sims", shortName, "benchmark.bson")
  workdir = datadir("sims", shortName, "temp-OMSimulator")

  pathToFMU = datadir("sims", shortName, "fmus", modelName*".onnx.fmu")
  resultFile = datadir("exp_raw", shortName, shortName * "_res.onnx.csv")
  logFile = joinpath(workdir, "OMSimulator_calls.onnx.log")
  bench = simulateFMU_OMSimulator(pathToFMU, resultFile; workdir=workdir, logFile=logFile, samples=1)
  @save benchmarkFile bench

  pathToFMU = datadir("sims", shortName, "fmus", modelName*".fmu")
  resultFile = datadir("exp_raw", shortName, shortName * "_res.csv")
  logFile = joinpath(workdir, "OMSimulator_calls.log")
  simulateFMU_OMSimulator(pathToFMU, resultFile; workdir=workdir, logFile=logFile, samples=1)
end
