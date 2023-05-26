# $ julia --threads=auto -e "include(\"runAll.jl\")"
# $ nohup julia --threads=auto -e "include(\"runAll.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

sizes = [5]

#begin
#  include("genAllSurrogates.jl")
#  #ENV["JULIA_DEBUG"] = NonLinearSystemNeuralNetworkFMU
#  genAllSurrogates(sizes, modelicaLib; n=1000, genData=false)
#end

#begins
#  include("simulateSurrogates.jl")
#  simulateAllSurrogates(sizes)
#end

# Generate plots
#sizes = [5, 10, 20, 40]
#begin
#  include("genAllPlots.jl")
#  include("plotTrainData.jl")
#  plotAllResults(sizes)
#  plotAllTrainingData(sizes)
#  simulationTimes(sizes; printAbsTime=false, plotTimeLabels=true, filename = plotsdir("ScalableTranslationStatistics.simTimeOverview.pdf"), title="")
#  plotTrainingProgress(sizes, format="pdf")
#  plotTrainingProgress(sizes, format="svg")
#end
