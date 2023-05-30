# $ julia --threads=auto -e "include(\"runAll.jl\")"
# $ nohup julia --threads=auto -e "include(\"runAll.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

sizes = [5]

begin
  include("genAllSurrogates.jl")
  #ENV["JULIA_DEBUG"] = NonLinearSystemNeuralNetworkFMU
  genAllSurrogates(sizes, modelicaLib; n=5000, genData=false)
end

begin
  include("simulateSurrogates.jl")
  simulateAllSurrogates(sizes)
end

# Generate plots
#sizes = [5,10,20,40]
begin
  include("genAllPlots.jl")
  include("plotTrainData.jl")
  plotAllResults(sizes) # TODO: Fix me!
  plotItterationVariables(sizes, filetype="pdf")
  plotAllTrainingData(sizes, filetype="pdf")
  simulationTimes(sizes; printAbsTime=false, plotTimeLabels=true, filename = plotsdir("ScalableTranslationStatistics.simTimeOverview.pdf"), title="")
  plotTrainingProgress(sizes, filetype="pdf")
  include("plotSimTimeOverview.jl")
  csvFile = "$(@__DIR__)/../simTimes.csv"
  #plotSimTimes(sizes, csvFile; filetype="pdf")
end

