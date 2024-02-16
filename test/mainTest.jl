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

function runMainTest()
  modelName = "simpleLoop"
  moFiles = [abspath(@__DIR__,"simpleLoop.mo")]
  workingDir = joinpath(abspath(@__DIR__), modelName)
  omOptions = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)
  dataGenOptions = NonLinearSystemNeuralNetworkFMU.DataGenOptions(method=NonLinearSystemNeuralNetworkFMU.RandomMethod(), n=10, nBatches=Threads.nthreads())
  rm(joinpath(workingDir), recursive=true, force=true)

  @testset "Generate Data (main)" begin
    (csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=false)
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
    usingVars = NonLinearSystemNeuralNetworkFMU.getUsingVars(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test sort(usingVars) == ["r","s"]
    @test length(usingVars) == 2
  end

  @testset "Test function getIterationVars" begin
    iterationVars = NonLinearSystemNeuralNetworkFMU.getIterationVars(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test iterationVars == ["y"]
    @test length(iterationVars) == 1
  end

  @testset "Test function getInnerEquations" begin
    innerEq = NonLinearSystemNeuralNetworkFMU.getInnerEquations(joinpath(workingDir, "profilingInfo.bson"), 14)
    @test sort(innerEq) == [11]
    @test length(innerEq) == 1
  end

  @testset "Test function getMinMax" begin
    minMax  = NonLinearSystemNeuralNetworkFMU.getMinMax(joinpath(workingDir, "profilingInfo.bson"), 14, ["r","s"])
    @test sort(minMax) ==  [[0.0, 1.4087228258248679], [0.95, 3.15]]
  end
  rm(joinpath(workingDir), recursive=true, force=true)
end

runMainTest()
