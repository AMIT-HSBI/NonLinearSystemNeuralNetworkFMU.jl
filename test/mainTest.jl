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

function runMainTest()
  modelName = "simpleLoop"
  moFiles = [abspath(@__DIR__,"simpleLoop.mo")]
  workingDir = joinpath(abspath(@__DIR__), modelName)
  options = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)
  rm(joinpath(workingDir), recursive=true, force=true)

  @testset "Generate Data (main)" begin
    (csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; options=options, reuseArtifacts=false, N=10)
    @test isfile(joinpath(workingDir, "profilingInfo.bson"))
    @test isfile(joinpath(workingDir, "simpleLoop.fmu"))
    @test isfile(joinpath(workingDir, "simpleLoop.interface.fmu"))
    @test length(csvFiles) == 1
    @test isfile(csvFiles[1])
    @test isfile(fmu)
    @test length(profilingInfo) == 1
    # NLS from simulation system
    @test profilingInfo[1].eqInfo.id == 14
    @test profilingInfo[1].iterationVariables == ["y"]
    @test sort(profilingInfo[1].usingVars) == ["r","s"]
    @test profilingInfo[1].boundary.min[1] ≈ 0.0 && profilingInfo[1].boundary.max[1] ≈ 1.4087228258248679
    @test profilingInfo[1].boundary.min[2] ≈ 0.95 && profilingInfo[1].boundary.max[2] ≈ 3.15
  end

  @testset "Test function getUsingVars" begin
    usingVars, lenUsingVars  = NonLinearSystemNeuralNetworkFMU.getUsingVars(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test sort(usingVars) == ["r","s"]
    @test lenUsingVars == 2
  end

  @testset "Test function getIterationVars" begin
    iterationVars, lenIterationVars  = NonLinearSystemNeuralNetworkFMU.getIterationVariables(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test iterationVars == ["y"]
    @test lenIterationVars == 1
  end

  @testset "Test function getInnerEquations" begin
    innerEq, lenInnerEq  = NonLinearSystemNeuralNetworkFMU.getInnerEquations(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test sort(innerEq) == [11]
    @test lenInnerEq == 1
  end
  rm(joinpath(workingDir), recursive=true, force=true)
end

runMainTest()
