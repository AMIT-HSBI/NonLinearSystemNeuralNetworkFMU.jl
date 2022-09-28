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

  @testset "Generate Data (main)" begin
    (csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; workdir=workingDir, reuseArtifacts=false, N=10)
    @test isfile(joinpath(workingDir, "minMax.bson"))
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
    rm(joinpath(workingDir), recursive=true, force=true)
  end
end

runMainTest()
