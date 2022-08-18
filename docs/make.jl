using Documenter, NonLinearSystemNeuralNetworkFMU

makedocs(
  sitename="NonLinearSystemNeuralNetworkFMU.jl",
  workdir=joinpath(@__DIR__,".."),
  pages = [
    "Home" => "index.md"
    "Profiling" => "profiling.md"
  ]
)

deploydocs(
  repo = "github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git",
  devbranch = "main"
)
