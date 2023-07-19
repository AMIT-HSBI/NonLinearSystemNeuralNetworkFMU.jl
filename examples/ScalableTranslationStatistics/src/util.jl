
function getNames(size)
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
  shortName = split(modelName, ".")[end]
  return (shortName, modelName)
end
