#
# Copyright (c) 2022-2023 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
#

using Test
import NonLinearSystemNeuralNetworkFMU

function runProfilingTests()
  modelName = "simpleLoop"
  moFiles = [abspath(@__DIR__,"simpleLoop.mo")]
  workingDir = joinpath(abspath(@__DIR__), modelName)
  options = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)

  @testset "Find slowes equations" begin
    profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, moFiles; options=options, threshold=0, ignoreInit=false)
    @test length(profilingInfo) == 2
    # NLS from simulation system
    @test profilingInfo[1].eqInfo.id == 14
    @test profilingInfo[1].iterationVariables == ["y"]
    @test sort(profilingInfo[1].usingVars) == ["r","s"]
    @test profilingInfo[1].boundary.min[1] ≈ 0.0 && profilingInfo[1].boundary.max[1] ≈ 1.4087228258248679
    @test profilingInfo[1].boundary.min[2] ≈ 0.95 && profilingInfo[1].boundary.max[2] ≈ 3.15
    # NLS from initialization system
    @test profilingInfo[2].eqInfo.id == 6
    @test profilingInfo[2].iterationVariables == ["y"]
    @test sort(profilingInfo[2].usingVars) == ["r","s"]
    rm(joinpath(workingDir,modelName), recursive=true)
  end
end

runProfilingTests()
