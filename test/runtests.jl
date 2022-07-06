using Test
using NonLinearSystemNeuralNetworkFMU

include("profilingTests.jl")
include("genFmusTest.jl")
include("genDataTest.jl")

pathToOmc = ""
if Sys.iswindows()
  pathToOmc = string(strip(read(`where omc.exe`, String)))
else
  pathToOmc = string(strip(read(`which omc`, String)))
end
if !isfile(pathToOmc)
  error("omc not found")
else
  @info "Using omc: $pathToOmc"
end

pathToFmiHeader = abspath(joinpath(@__DIR__, "..", "FMI-Standard-2.0.3","headers"))

runProfilingTests()
runGenFmusTest()
runGenDataTest()
