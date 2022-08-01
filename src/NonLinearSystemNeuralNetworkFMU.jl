#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

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
import ZipFile

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
