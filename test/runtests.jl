#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

using SafeTestsets

@safetestset "Profiling" begin include("profilingTests.jl") end
@safetestset "Generate FMUs" begin include("genFmusTest.jl") end
@safetestset "Generate data" begin include("genDataTest.jl") end
@safetestset "Train ANN" begin include("trainNNTest.jl") end
@safetestset "Generate ONNX FMU" begin include("includeOnnxTest.jl") end
