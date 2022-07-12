
function runGenFmusTest()
  @testset "FMU generation test" begin
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
      pathToFmu = NonLinearSystemNeuralNetworkFMU.generateFMU(modelName = modelName,
                                                              pathToMo = pathToMo,
                                                              pathToOmc = pathToOmc,
                                                              tempDir = workingDir)
      @test isfile(pathToFmu)
      # Save FMU for next test
      cp(pathToFmu, joinpath(fmuDir, modelName*".fmu"))
      rm(workingDir, recursive=true)
    end

    @testset "Re-compile extended FMU" begin
      pathToFmu = abspath(joinpath(fmuDir, "$(modelName).fmu"))
      eqIndices = [14]
      tempDir = abspath(joinpath(fmuDir, "$(modelName)_interface"))
      if !isdir(tempDir)
        mkdir(tempDir)
      else
        rm(tempDir, force=true, recursive=true)
        mkdir(tempDir)
      end
      pathToFmu = NonLinearSystemNeuralNetworkFMU.addEqInterface2FMU(modelName = modelName,
                                                                     pathToFmu = pathToFmu,
                                                                     pathToFmiHeader = pathToFmiHeader,
                                                                     eqIndices = eqIndices,
                                                                     tempDir = tempDir)
      @test isfile(pathToFmu)
      # Save FMU for next test
      cp(pathToFmu, joinpath(fmuDir, modelName*".interface.fmu"))
      rm(tempDir, recursive=true)
    end
  end
end
