using DrWatson
@quickactivate "ScalableTranslationStatistics"

include(srcdir("plotResult.jl"))

sizes = [5, 10, 20, 40, 80]

outputVars = ["outputs[1]", "outputs[2]", "outputs[3]", "outputs[4]", "outputs[5]", "outputs[6]", "outputs[7]", "outputs[8]"]

for size in sizes
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
  shortName = split(modelName, ".")[end]

  refResult = datadir("exp_raw", shortName, shortName * "_res.csv")
  surrogateResult = datadir("exp_raw", shortName, shortName * "_res.onnx.csv")

  fig = plotResult(refResult, surrogateResult, outputVars; tspan=(0.0, 10.0))

  mkpath(dirname(plotsdir(modelName, "$(shortName)_results.svg")))
  save(plotsdir(modelName, "$(shortName)_results.svg"), fig)
end
