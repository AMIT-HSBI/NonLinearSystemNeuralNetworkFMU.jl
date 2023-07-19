using DrWatson
@quickactivate "ScalableTranslationStatistics"

using BSON: @save

include(srcdir("simulateFMU.jl"))

function simulateAllSurrogates(sizes)
  for size in sizes
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
    shortName = split(modelName, ".")[end]

    benchmarkFile = datadir("sims", shortName, "benchmark.bson")
    workdir = datadir("sims", shortName, "temp-OMSimulator")

    pathToFMU = datadir("sims", shortName, "fmus", modelName*".onnx.fmu")
    resultFile = datadir("exp_raw", shortName, shortName * "_res.onnx.csv")
    logFile = joinpath(workdir, "OMSimulator_calls.onnx.log")
    luaFile = joinpath(workdir, "$(modelName)_onnx.lua")
    bench = simulateFMU_OMSimulator(luaFile, pathToFMU, shortName, resultFile; workdir=workdir, logFile=logFile, samples=1)
    @save benchmarkFile bench

    pathToFMU = datadir("sims", shortName, "fmus", modelName*".fmu")
    resultFile = datadir("exp_raw", shortName, shortName * "_res.csv")
    logFile = joinpath(workdir, "OMSimulator_calls.log")
    luaFile = joinpath(workdir, "$(modelName)_ref.lua")
    simulateFMU_OMSimulator(luaFile, pathToFMU, shortName, resultFile; workdir=workdir, logFile=logFile, samples=1)
  end
end
