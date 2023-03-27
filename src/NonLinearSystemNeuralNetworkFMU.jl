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

import BSON
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
import Suppressor
import XMLDict

# This symbol is only defined on Julia versions that support extensions
if !isdefined(Base, :get_extension)
  import Requires
end

include("types.jl")
export EqInfo
export ProfilingInfo
export OMOptions
export getUsingVars
export getIterationVars
export getInnerEquations
export getMinMax
include("util.jl")
include("profiling.jl")
export profiling
export minMaxValuesReSim
include("genFMUs.jl")
export generateFMU
export addEqInterface2FMU
include("genTrainData.jl")
export generateTrainingData
include("integrateNN.jl")
export buildWithOnnx
include("main.jl")
export main

"""
    plotTrainArea()
"""
function plotTrainArea()
  @error "Load CairoMakie before using this"
  @info "Usage: plotTrainArea(vars, df_ref; df_surrogate=nothing, df_trainData=nothing, title=\"\", epsilon=0.01, tspan=nothing)"
end
export plotTrainArea

function __init__()
  @static if !isdefined(Base, :get_extension)
    Requires.@require CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0" begin
      @eval include(normpath(@__DIR__, "..", "ext", "PlottingMakieExt.jl"))
    end
  end
end

end # module
