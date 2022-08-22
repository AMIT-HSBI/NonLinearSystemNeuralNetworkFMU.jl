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
import NonLinearSystemNeuralNetworkFMU

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

function runProfilingTests()
  modelName = "simpleLoop"
  pathToMo = abspath(@__DIR__,"simpleLoop.mo")
  workingDir = abspath(@__DIR__)

  @testset "Find slowes equations" begin
    profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc; workingDir=workingDir, threshold=0)
    @test length(profilingInfo) == 2
    # NLS from simulation system
    @test profilingInfo[1].eqInfo.id == 14
    @test profilingInfo[1].iterationVariables == ["y"]
    @test sort(profilingInfo[1].usingVars) == ["r","s"]
    # NLS from initialization system
    @test profilingInfo[2].eqInfo.id == 6
    @test profilingInfo[2].iterationVariables == ["y"]
    @test sort(profilingInfo[2].usingVars) == ["r","s"]
    rm(joinpath(workingDir,modelName), recursive=true)
  end

  @testset "Min-max for usingVars" begin
    profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc; workingDir=workingDir, threshold=0)
    (min, max)  = NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim(profilingInfo[1].usingVars, modelName, pathToMo, pathToOmc; workingDir=workingDir)
    # s = sqrt((2-time)*0.9), time = 0..2
    # r = 1..3
    @test min[1] ≈ 0.0 && max[1] ≈ 1.4087228258248679
    @test min[2] ≈ 0.95 && max[2] ≈ 3.15
    rm(joinpath(workingDir,modelName), recursive=true)
  end
end

runProfilingTests()
