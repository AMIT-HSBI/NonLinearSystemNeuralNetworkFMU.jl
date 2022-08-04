#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using Test
using NonLinearSystemNeuralNetworkFMU

include("profilingTests.jl")
include("genFmusTest.jl")
include("genDataTest.jl")
include("trainNNTest.jl")

pathToOmc = ""
if Sys.iswindows()
  @assert(haskey(ENV, "OPENMODELICAHOME"), "Environment variable OPENMODELICAHOME not set.")
  pathToOmc = abspath(joinpath(ENV["OPENMODELICAHOME"], "bin", "omc.exe"))
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
runTrainNNTest()
