module NonLinearSystemNeuralNetworkFMU

import Printf
import OMJulia
import JSON
import CSV
import DataFrames

include("types.jl")
export EqInfo
export ProfilingInfo
include("profiling.jl")
export profiling
include("genTrainData.jl")
export generateTrainingData

export generateFMUX

end # module
