using Documenter, NonLinearSystemNeuralNetworkFMU

ENV["JULIA_DEBUG"]="Documenter"

@info "Make the docs"
makedocs(
  sitename = "NonLinearSystemNeuralNetworkFMU.jl",
  format = Documenter.HTML(edit_link = "main"),
  workdir = joinpath(@__DIR__,".."),
  pages = [
    "Home" => "index.md",
    "Main" => "main.md",
    "Profiling" => "profiling.md",
    "Data Generation" => "dataGen.md",
    "ONNX Generation" => "train.md",
    "Integrate ONNX" => "integrateONNX.md"
  ]
)

#@info "Deploy the docs"
#deploydocs(
#  repo = "github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git",
#  devbranch = "main"
#)
