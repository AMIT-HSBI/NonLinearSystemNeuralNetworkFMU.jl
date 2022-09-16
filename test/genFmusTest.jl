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

function runGenFmusTest()
  modelName = "simpleLoop"
  pathToMo = abspath(@__DIR__,"simpleLoop.mo")
  workingDir = abspath(joinpath(@__DIR__, modelName))
  fmuDir = abspath(joinpath(@__DIR__, "fmus"))

  if !isdir(fmuDir)
    mkdir(fmuDir)
  else
    rm(fmuDir, force=true, recursive=true)
    mkdir(fmuDir)
  end

  @testset "Generate default FMU" begin
    pathToFmu = NonLinearSystemNeuralNetworkFMU.generateFMU(modelName, pathToMo, workingDir = workingDir)
    @test isfile(pathToFmu)
    # Save FMU for next test
    cp(pathToFmu, joinpath(fmuDir, modelName*".fmu"))
    rm(workingDir, recursive=true)
  end

  @testset "Re-compile extended FMU" begin
    pathToFmu = abspath(joinpath(fmuDir, "$(modelName).fmu"))
    eqIndices = [14]
    workingDir = abspath(joinpath(fmuDir, "$(modelName)_interface"))
    if !isdir(workingDir)
      mkdir(workingDir)
    else
      rm(workingDir, force=true, recursive=true)
      mkdir(workingDir)
    end
    pathToFmu = NonLinearSystemNeuralNetworkFMU.addEqInterface2FMU(modelName, pathToFmu, eqIndices, workingDir = workingDir)
    @test isfile(pathToFmu)
    # Save FMU for next test
    cp(pathToFmu, joinpath(fmuDir, modelName*".interface.fmu"))
    rm(workingDir, recursive=true)
  end
end

runGenFmusTest()
