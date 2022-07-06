
function runProfilingTests()
  @testset "Profiling test" begin
    model_name = "simpleLoop"
    path_to_mo = abspath(@__DIR__,"simpleLoop.mo")
    working_dir = abspath(@__DIR__)

    @testset "Find slowes equations" begin
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(model_name, path_to_mo, path_to_omc, working_dir)
      @test length(profilingInfo) == 1
      @test profilingInfo[1].eqInfo.id == 14
      @test profilingInfo[1].iterationVariables == ["y"]
      @test sort(profilingInfo[1].usingVars) == ["r","s"]
      rm(joinpath(working_dir,model_name), recursive=true)
    end

    @testset "Min-max for usingVars" begin
      profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(model_name, path_to_mo, path_to_omc, working_dir)
      (min, max)  = NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim(profilingInfo[1].usingVars, model_name, path_to_mo, path_to_omc, working_dir)
      # s = sqrt((2-time)*0.9), time = 0..2
      # r = 1..3
      @test min[1] ≈ 0.0 && max[1] ≈ 1.4087228258248679
      @test min[2] ≈ 0.95 && max[2] ≈ 3.15
      rm(joinpath(working_dir,model_name), recursive=true)
    end
  end
end
