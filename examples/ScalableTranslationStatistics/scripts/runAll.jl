# $ julia --threads=auto -e "include(\"runAll.jl\")"
# $ nohup julia --threads=auto -e "include(\"runAll.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

#sizes = [5, 10, 20, 40, 80]
sizes = [5, 10, 20,40]

begin
  #include("genAllSurrogates.jl")
  #genAllSurrogates(sizes, modelicaLib; N=1000)
end

begin
  #include("simulateSurrogates.jl")
  #simulateAllSurrogates(sizes)
end

# Generate plots
begin
  #include("genAllPlots.jl")
  #plotAllResults(sizes)
  include("plotTrainData.jl")
  plotAllTrainingData(sizes)
  simulationTimes(sizes; printAbsTime=true)
end
