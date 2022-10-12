using Documenter, NonLinearSystemNeuralNetworkFMU

@info "Make the docs"
makedocs(
  sitename="NonLinearSystemNeuralNetworkFMU.jl",
  format = Documenter.LaTeX(platform = "docker"),   # Workaround because we can't publish HTML to GitHub pages at the moment
  workdir=joinpath(@__DIR__,".."),
  pages = [
    "Home" => "index.md",
    "Main" => "main.md",
    "Profiling" => "profiling.md",
    "Data Generation" => "dataGen.md"
  ]
)

@info "Deploy the docs"
deploydocs(
  repo = "github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git",
  devbranch = "main"
)
