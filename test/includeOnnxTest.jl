#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using Revise
using Test
using NonLinearSystemNeuralNetworkFMU

modelName = "simpleLoop"
fmuroot = abspath(@__DIR__,"fmus/FMU/")
profilingInfo = ProfilingInfo[ProfilingInfo(EqInfo(14, 2512, 2.111228e6, 54532.0, 0.12241628639186376), ["y"], [11, 12, 13], ["s", "r"]), ProfilingInfo(EqInfo(6, 8, 30248.0, 30374.0, 0.0017538834416657486), ["y"], [11, 12, 13], ["s", "r"])]
modelDescription = abspath(@__DIR__,"fmus/FMU/modelDescription.xml")
ortdir = "/mnt/home/aheuermann/workdir/julia/benchmark-import-NN/onnxruntime-linux-x64-1.11.0"
onnxFiles = [abspath(@__DIR__, "nn", "simpleLoop_eq14.onnx")]
NonLinearSystemNeuralNetworkFMU.buildWithOnnx(modelName, fmuroot, [profilingInfo[1]], onnxFiles, ortdir)