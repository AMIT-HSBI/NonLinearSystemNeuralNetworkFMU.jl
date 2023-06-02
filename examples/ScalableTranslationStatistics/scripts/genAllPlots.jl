using DrWatson
@quickactivate "ScalableTranslationStatistics"

using NonLinearSystemNeuralNetworkFMU

include(srcdir("plotResult.jl"))

function plotAllResults(sizes; filetype="pdf", plotAbsErr::Bool=true)
  for size in sizes
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
    shortName = split(modelName, ".")[end]
    outputVars = "$(shortName).scalableModelicaModel." .* ["outputs[1]", "outputs[2]", "outputs[3]", "outputs[4]", "outputs[5]", "outputs[6]", "outputs[7]", "outputs[8]"]

    refResult = datadir("exp_raw", shortName, shortName * "_res.csv")
    surrogateResult = datadir("exp_raw", shortName, shortName * "_res.onnx.csv")

    @info "Plotting results for size $size"
    fig = plotResult(refResult, surrogateResult, outputVars[1:3], "Output"; plotAbsErr=plotAbsErr, tspan=(0.0, 10.0))

    fileName = plotsdir(shortName, "$(shortName)_results.$(filetype)")
    mkpath(dirname(fileName))
    save(fileName, fig)
  end
end

function plotItterationVariables(sizes; filetype="pdf")
  for size in sizes
    @info "Plot itteration variables for N=$size"
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
    shortName = split(modelName, ".")[end]

    profilingInfo = getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))
    for prof in profilingInfo
      outputVars = "$(shortName)." .* prof.iterationVariables

      refResult = datadir("exp_raw", shortName, shortName * "_res.csv")
      surrogateResult = datadir("exp_raw", shortName, shortName * "_res.onnx.csv")
      residualResults = datadir("sims", shortName, "temp-OMSimulator", "$(modelName)_eq$(prof.eqInfo.id)_residuum.csv")

      @info "Plotting itteration variables for eq $(prof.eqInfo.id)"
      fig1, fig2 = plotResult(refResult, surrogateResult, outputVars, "Iteration"; tspan=(0.0, 10.0), residualResults=residualResults, fullNames=true, eqId=prof.eqInfo.id)

      fileName = plotsdir(shortName, "loop_$(prof.eqInfo.id)_results_a.$(filetype)")
      mkpath(dirname(fileName))
      save(fileName,fig1)
      fileName = plotsdir(shortName, "loop_$(prof.eqInfo.id)_results_b.$(filetype)")
      save(fileName,fig2)
    end
  end
end
