
function runProfilingTests()
  @testset "Profiling test" begin
    modelName = "simpleLoop"
    pathToMo = abspath(@__DIR__,"simpleLoop.mo")
    workingDir = abspath(@__DIR__)

    @testset "Find slowes equations" begin
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc, workingDir)
      @test length(profilingInfo) == 1
      @test profilingInfo[1].eqInfo.id == 14
      @test profilingInfo[1].iterationVariables == ["y"]
      @test sort(profilingInfo[1].usingVars) == ["r","s"]
      rm(joinpath(workingDir,modelName), recursive=true)
    end

    @testset "Min-max for usingVars" begin
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc, workingDir)
      (min, max)  = NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim(profilingInfo[1].usingVars, modelName, pathToMo, pathToOmc, workingDir)
      # s = sqrt((2-time)*0.9), time = 0..2
      # r = 1..3
      @test min[1] ≈ 0.0 && max[1] ≈ 1.4087228258248679
      @test min[2] ≈ 0.95 && max[2] ≈ 3.15
      rm(joinpath(workingDir,modelName), recursive=true)
    end
  end
end
