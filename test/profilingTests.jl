#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

function runProfilingTests()
  @testset "Profiling test" begin
    modelName = "simpleLoop"
    pathToMo = abspath(@__DIR__,"simpleLoop.mo")
    workingDir = abspath(@__DIR__)

    @testset "Find slowes equations" begin
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc, workingDir; threshold=0)
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
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc, workingDir; threshold=0)
      (min, max)  = NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim(profilingInfo[1].usingVars, modelName, pathToMo, pathToOmc, workingDir)
      # s = sqrt((2-time)*0.9), time = 0..2
      # r = 1..3
      @test min[1] ≈ 0.0 && max[1] ≈ 1.4087228258248679
      @test min[2] ≈ 0.95 && max[2] ≈ 3.15
      rm(joinpath(workingDir,modelName), recursive=true)
    end
  end
end
