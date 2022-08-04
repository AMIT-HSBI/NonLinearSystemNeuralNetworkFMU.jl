#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using Test
import NonLinearSystemNeuralNetworkFMU

pathToOmc = ""
if Sys.iswindows()
  @assert(haskey(ENV, "OPENMODELICAHOME"), "Environment variable OPENMODELICAHOME not set.")
  pathToOmc = abspath(joinpath(ENV["OPENMODELICAHOME"], "bin", "omc.exe"))
else
  pathToOmc = string(strip(read(`which omc`, String)))
end
if !isfile(pathToOmc)
  error("omc not found")
else
  @info "Using omc: $pathToOmc"
end

pathToFmiHeader = abspath(joinpath(@__DIR__, "..", "FMI-Standard-2.0.3","headers"))

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

runGenFmusTest()
