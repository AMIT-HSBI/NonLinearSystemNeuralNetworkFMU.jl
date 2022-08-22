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

module NonLinearSystemNeuralNetworkFMU

import CSV
import DataFrames
import DocStringExtensions
import FMI
import FMICore
import FMIImport
import JSON
import Libdl
import OMJulia
import Printf
import ProgressMeter
import XMLDict

include("types.jl")
export EqInfo
export ProfilingInfo
include("profiling.jl")
export profiling
export minMaxValuesReSim
include("genFMUs.jl")
export generateFMU
export addEqInterface2FMU
include("genTrainData.jl")
export generateTrainingData
include("integrateNN.jl")

end # module
