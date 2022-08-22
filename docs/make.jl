using Documenter, NonLinearSystemNeuralNetworkFMU

makedocs(
  sitename="NonLinearSystemNeuralNetworkFMU.jl",
  format = Documenter.LaTeX(platform = "docker"),   # Workaround because we can't publish HTML to GitHub pages at the moment
  workdir=joinpath(@__DIR__,".."),
  pages = [
    "Home" => "index.md",
    "Profiling" => "profiling.md",
    "Data Generation" => "dataGen.md"
  ]
)

deploydocs(
  repo = "github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git",
  devbranch = "main"
)
