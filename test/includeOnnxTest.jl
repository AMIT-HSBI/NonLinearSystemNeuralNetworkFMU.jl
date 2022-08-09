#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using Test
using FMI
using NonLinearSystemNeuralNetworkFMU

function runIncludeOnnxTests()
  @testset "Build FMU with ONNX" begin
    modelname = "simpleLoop"
    fmuDir = abspath(joinpath(@__DIR__, "fmus"))
    tempDir = joinpath(fmuDir, "$(modelname)_onnx")
    interfaceFmu = joinpath(fmuDir, "$(modelname).interface.fmu")
    onnxFmu = joinpath(fmuDir, "$(modelname).onnx.fmu")
    rm(onnxFmu, force=true)
    profilingInfo = ProfilingInfo[ProfilingInfo(EqInfo(14, 2512, 2.111228e6, 54532.0, 0.12241628639186376), ["y"], [11], ["s", "r"])]
    ortdir = "/mnt/home/aheuermann/workdir/julia/benchmark-import-NN/onnxruntime-linux-x64-1.11.0"
    onnxFiles = [abspath(@__DIR__, "nn", "simpleLoop_eq14.onnx")]

    pathToFmu = NonLinearSystemNeuralNetworkFMU.buildWithOnnx(interfaceFmu, modelname, profilingInfo, onnxFiles, ortdir; tempDir=tempDir)
    @test isfile(pathToFmu)
    # Save FMU for next test
    cp(pathToFmu, onnxFmu)
    #rm(tempDir, recursive=true)
  end

  @testset "Simulate ONNX FMU" begin
    #modelname = "simpleLoop"
    #pathToFMU = abspath(joinpath(@__DIR__, "fmus", "$(modelname).onnx.fmu"))
    #myFMU = fmiLoad(pathToFMU)
    #res = fmiSimulate(myFMU, 0.0, 1.0; recordValues=["r", "s", "x", "y"])

  end
end

runIncludeOnnxTests()
