# $ julia --threads=auto -e "include(\"runAll.jl\")"
# $ nohup julia --threads=auto -e "include(\"runAll.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

sizes = [40]

begin
  include("genAllSurrogates.jl")
  #ENV["JULIA_DEBUG"] = NonLinearSystemNeuralNetworkFMU
  genAllSurrogates(sizes, modelicaLib; n=5000, genData=true)
  logFile = plotsdir("LoopInfo.log")
  logProfilingInfo(sizes, logFile)
end

begin
  include("simulateSurrogates.jl")
  simulateAllSurrogates(sizes)
end

# Generate plots
sizes = [5,10,20,40]
begin
  include("genAllPlots.jl")
  include("plotTrainData.jl")
  plotAllResults(sizes, plotAbsErr=false, filetype="png")
  plotItterationVariables(sizes, filetype="png")
  plotAllTrainingData(sizes, filetype="png")
  simulationTimes(sizes; printAbsTime=false, plotTimeLabels=true, filetype="png", title="")
  plotTrainingProgress(sizes, filetype="png")
  csvFile = "$(@__DIR__)/../simTimes.csv"
  include("plotSimTimeOverview.jl")
  plotSimTimes(sizes, csvFile; filetype="png")
end

