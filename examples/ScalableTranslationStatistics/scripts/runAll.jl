# $ julia --threads=auto -e "include(\"runAll.jl\")"
# $ nohup julia --threads=auto -e "include(\"runAll.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

begin
  include("genAllSurrogates.jl")
end

begin
  include("simulateSurrogates.jl")
end

# Generate plots
begin
  include("genAllPlots.jl")
  include("plotTrainData.jl")
end
