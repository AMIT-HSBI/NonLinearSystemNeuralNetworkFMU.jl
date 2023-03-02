# $ julia --threads=auto -e "include(\"runall.jl\")"
# $ nohup julia --threads=auto -e "include(\"runall.jl\")" &

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
end
