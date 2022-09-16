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

function runGenDataTest()
  pathToFMU = abspath(joinpath(@__DIR__, "fmus", "simpleLoop.interface.fmu"))
  eqIndex = 14
  inputVars = ["s", "r"]
  outputVars = ["y"]
  min = [0.0, 0.95]
  max = [1.5, 3.15]
  fileName = abspath(joinpath(@__DIR__, "data", "simpleLoop_eq14.csv"))
  N = 100

  NonLinearSystemNeuralNetworkFMU.generateTrainingData(pathToFMU, fileName,
                                                       eqIndex, inputVars,
                                                       min, max, outputVars;
                                                       N = N)

  @test isfile(fileName)
  nLines = 0
  # Check if s,r,y solve algebraic loop
  open(fileName,"r") do f
    @test readline(f) === "s,r,y"
    isequal = true
    while !eof(f) && isequal
      nLines += 1
      line = readline(f)
      s,r,y = parse.(Float64,split(line,","))
      x = r*s -y
      isequal = r^2 ≈ x^2 + y^2
    end
    @test isequal
  end
  # At least 80% successfull data generation
  @test nLines >= N*0.8 && nLines <= N
end

runGenDataTest()
