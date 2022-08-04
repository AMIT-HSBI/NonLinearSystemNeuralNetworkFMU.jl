#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using Revise
using Test
using NonLinearSystemNeuralNetworkFMU

pathToOmc = string(strip(read(`which omc`, String)))
modelName = "simpleLoop"
pathToMo = abspath(@__DIR__,"simpleLoop.mo")
workingDir = abspath(@__DIR__)
simpleLoopFile = abspath(@__DIR__,"fmus/FMU/sources/simpleLoop.c")

profilingInfo = NonLinearSystemNeuralNetworkFMU.profiling(modelName, pathToMo, pathToOmc, workingDir; threshold=0)

modelDescription = abspath(@__DIR__,"fmus/FMU/modelDescription.xml")
NonLinearSystemNeuralNetworkFMU.addNNCall(modelName, simpleLoopFile, modelDescription, profilingInfo[1])