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

import CSV
import DataFrames
using FMI
using NonLinearSystemNeuralNetworkFMU
using Test

function runIncludeOnnxTests()
  @assert haskey(ENV, "ORT_DIR") "Environamet variable `ORT_DIR` has to be set and point to ONNX Runtime directory for testing."

  @testset "Build FMU with ONNX" begin
    modelname = "simpleLoop"
    fmuDir = abspath(joinpath(@__DIR__, "fmus"))
    tempDir = joinpath(fmuDir, "$(modelname)_onnx")
    rm(tempDir, force=true, recursive=true)
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
  end

  @testset "Simulate ONNX FMU" begin
    modelname = "simpleLoop"
    workDir = joinpath(@__DIR__, "oms")
    rm(workDir, force=true, recursive=true)
    mkpath(workDir)

    fmuDir = joinpath(@__DIR__, "fmus")
    fmu = joinpath(fmuDir, "$(modelname).onnx.fmu")

    resultFile = "simpleLoop_onnx_res.csv"
    logFile = joinpath(workDir, modelname*"_OMSimulator.log")

    cmd = `OMSimulator --resultFile=$(resultFile) "$(fmu)"`
    NonLinearSystemNeuralNetworkFMU.omrun(cmd, dir=workDir, logFile=logFile, timeout=60)

    @test isfile(joinpath(workDir, resultFile))
    @test read(logFile, String) == """
    info:    model doesn't contain any continuous state
    info:    Result file: simpleLoop_onnx_res.csv (bufferSize=1)
    """

    @test isfile(joinpath(workDir, "$(resultFile)"))
  end

  @testset "Check results" begin
    resultFile = joinpath(@__DIR__, "oms", "simpleLoop_onnx_res.csv")
    df_res = CSV.read(resultFile, DataFrames.DataFrame; ntasks=1)
    # LOG_RES is true, so if the result is too bad solve_nonlinear_system() is called
    # So the results should be very good.
    x = df_res.r .* df_res.s .- df_res.y
    @test maximum(abs.(df_res.r .^ 2 .- (x .^ 2 .+ df_res.y .^ 2))) < 1e-6
  end
end

runIncludeOnnxTests()
