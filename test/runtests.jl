#
# Copyright (c) 2022 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
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
