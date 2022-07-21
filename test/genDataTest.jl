#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

function runGenDataTest()
  @testset "Training data generation test" begin
    pathToFMU = abspath(joinpath(@__DIR__, "fmus", "simpleLoop.interface.fmu"))
    eqIndex = 14
    inputVars = ["s", "r"]
    outputVars = ["y"]
    min = [0.0, 0.95]
    max = [1.5, 3.15]
    fileName = abspath(joinpath(@__DIR__, "data", "simpleLoop_eq14.csv"))
    N = 3

    NonLinearSystemNeuralNetworkFMU.generateTrainingData(pathToFMU, fileName,
                                                         eqIndex, inputVars,
                                                         min, max, outputVars;
                                                         N = N)

    @test isfile(fileName)
    nLines = 0
    # Check if s,r,y solve algebraic loop
    open(fileName,"r") do f
      @test readline(f) === "s,r,y"
      while !eof(f)
        nLines += 1
        line = readline(f)
        s,r,y = parse.(Float64,split(line,","))
        x = r*s -y
        @test r^2 ≈ x^2 + y^2
      end
    end
    @test nLines == N
  end
end
