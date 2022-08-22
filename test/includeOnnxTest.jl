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

using Test
using FMI
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
    profilingInfo = ProfilingInfo[ProfilingInfo(EqInfo(14, 2512, 2.111228e6, 54532.0, 0.12241628639186376), ["y"], [11], ["s", "r"])]
    ortdir = ENV["ORT_DIR"]
    onnxFiles = [abspath(@__DIR__, "nn", "simpleLoop_eq14.onnx")]

    pathToFmu = NonLinearSystemNeuralNetworkFMU.buildWithOnnx(interfaceFmu, modelname, profilingInfo, onnxFiles, ortdir; tempDir=tempDir)
    @test isfile(pathToFmu)
    # Save FMU for next test
    cp(pathToFmu, onnxFmu)
    rm(tempDir, recursive=true)
  end

  @testset "Simulate ONNX FMU" begin
    modelname = "simpleLoop"
    pathToFMU = abspath(joinpath(@__DIR__, "fmus", "$(modelname).onnx.fmu"))
    nnFMU = fmiLoad(pathToFMU)
    # Can't use FMI.jl for FMUs without states at the moment
    #onnx_solution = fmiSimulate(nnFMU, 0.0, 1.0; recordValues=["r", "s", "x", "y"])
    fmiUnload(nnFMU)

    #pathToFMU = abspath(joinpath(@__DIR__, "fmus", "$(modelname).fmu"))
    #refFMU = fmiLoad(pathToFMU)
    #ref_solution = fmiSimulate(refFMU, 0.0, 1.0; recordValues=["r", "s", "x", "y"])
    #fmiUnload(refFMU)
  end
end

runIncludeOnnxTests()
