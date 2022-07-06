
function runGenFmusTest()
  @testset "FMU generation test" begin
    model_name = "simpleLoop"
    path_to_mo = abspath(@__DIR__,"simpleLoop.mo")
    working_dir = abspath(joinpath(@__DIR__, model_name))
    fmu_dir = abspath(joinpath(@__DIR__, "fmus"))

    if !isdir(fmu_dir)
      mkdir(fmu_dir)
    else
      rm(fmu_dir, force=true, recursive=true)
      mkdir(fmu_dir)
    end

    @testset "Generate default FMU" begin
      path_to_fmu = NonLinearSystemNeuralNetworkFMU.generateFMU(model_name = model_name,
                                                                path_to_mo = path_to_mo,
                                                                path_to_omc = path_to_omc,
                                                                tempDir = working_dir)
      @test isfile(path_to_fmu)
      # Save FMU for next test
      cp(path_to_fmu, joinpath(fmu_dir, model_name*".fmu"))
      rm(working_dir, recursive=true)
    end

    @testset "Re-compile extended FMU" begin
      path_to_fmu = abspath(joinpath(fmu_dir, "$(model_name).fmu"))
      eqIndices = [14]
      tempDir = abspath(joinpath(fmu_dir, "$(model_name)_interface"))
      if !isdir(tempDir)
        mkdir(tempDir)
      else
        rm(tempDir, force=true, recursive=true)
        mkdir(tempDir)
      end
      path_to_fmu = NonLinearSystemNeuralNetworkFMU.addEqInterface2FMU(model_name = model_name,
                                                                      path_to_fmu = path_to_fmu,
                                                                      path_to_fmi_header = path_to_fmi_header,
                                                                      eqIndices = eqIndices,
                                                                      tempDir = tempDir)
      @test isfile(path_to_fmu)
      # Save FMU for next test
      cp(path_to_fmu, joinpath(fmu_dir, model_name*".interface.fmu"))
      rm(tempDir, recursive=true)
    end
  end
end
