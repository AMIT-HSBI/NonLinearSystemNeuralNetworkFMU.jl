using DrWatson
@quickactivate "SimpleLoop_ActiveLearning"

import NonLinearSystemNeuralNetworkFMU

include(srcdir("main.jl"))
include(srcdir("activeLearn.jl"))
include(srcdir("simulateFMU.jl"))

modelName = "simpleLoop"
N = 1_000

mymain(modelName, N)
