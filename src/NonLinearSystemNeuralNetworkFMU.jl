module NonLinearSystemNeuralNetworkFMU

import Printf
import OMJulia
import JSON
import CSV
import DataFrames
import Libdl
import FMI
import FMICore
import FMIImport
import ProgressMeter

include("types.jl")
export EqInfo
export ProfilingInfo
include("profiling.jl")
export profiling
include("genFMUs.jl")
export generateFMU
export addEqInterface2FMU
include("genTrainData.jl")
export generateTrainingData

end # module
