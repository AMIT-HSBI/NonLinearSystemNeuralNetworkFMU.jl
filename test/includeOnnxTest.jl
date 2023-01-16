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

using FMI
using Suppressor
using Test
using NonLinearSystemNeuralNetworkFMU

function runIncludeOnnxTests()
  @assert haskey(ENV, "ORT_DIR") "Environamet variable `ORT_DIR` has to be set and point to ONNX Runtime directory for testing."

  @testset "Build FMU with ONNX" begin
    modelname = "simpleLoop"
    fmuDir = abspath(joinpath(@__DIR__, "fmus"))
    tempDir = joinpath(fmuDir, "$(modelname)_onnx")
    interfaceFmu = joinpath(fmuDir, "$(modelname).interface.fmu")
    onnxFmu = joinpath(fmuDir, "$(modelname).onnx.fmu")
    rm(onnxFmu, force=true)
    profilingInfo = ProfilingInfo[
      ProfilingInfo(
        EqInfo(14, 2512, 2.111228e6, 54532.0, 0.12241628639186376),
        ["y"],
        [11],
        ["s", "r"],
        NonLinearSystemNeuralNetworkFMU.MinMaxBoundaryValues([0.0, 0.95], [1.4087228258248679, 3.15]))]
    onnxFiles = [abspath(@__DIR__, "nn", "simpleLoop_eq14.onnx")]

    pathToFmu = NonLinearSystemNeuralNetworkFMU.buildWithOnnx(interfaceFmu, modelname, profilingInfo, onnxFiles; tempDir=tempDir)
    @test isfile(pathToFmu)
    # Save FMU for next test
    cp(pathToFmu, onnxFmu)
    rm(tempDir, recursive=true)
  end

  @testset "Simulate ONNX FMU" begin
    workDir = joinpath(@__DIR__, "fmus")
    modelname = "simpleLoop"
    resultFile = "model_onnx_res.csv"
    rm(joinpath(workDir,resultFile))
    logFile = joinpath(workDir, modelname*"_OMSimulator.log")

    cmd = `OMSimulator --resultFile=$resultFile "$(modelname).onnx.fmu"`
    redirect_stdio(stdout=logFile, stderr=logFile) do
      run(Cmd(cmd, dir=workDir))
    end

    @test isfile(joinpath(workDir,resultFile))
    @test read(logFile, String) == """
    info:    model doesn't contain any continuous state
    info:    Result file: model_onnx_res.csv (bufferSize=1)
    """
    rm(logFile)
  end

  @testset "Check results" begin
    resultFile = joinpath(@__DIR__, "fmus", "model_onnx_res.csv")
    df_res = CSV.read(resultFile, DataFrames.DataFrame; ntasks=1)
    @test maximum(abs.(df_res.x .- df_res.x_ref)) < 1e-2
    @test maximum(abs.(df_res.y .- df_res.y_ref)) < 1e-2
  end
end

runIncludeOnnxTests()
