#
# Copyright (c) 2022 Andreas Heuermann, Philip Hannebohm
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

include("fmiExtensions.jl")

"""
    generateTrainingData(fmuPath::String,
                         fname::String,
                         eqId::Int64,
                         inputVars::Array{String},
                         min::AbstractVector{<:Number}
                         max::AbstractVector{<:Number},
                         outputVars::Array{String};
                         N::Integer=1000)

Generate training data for given equation of FMU.

Generate random inputs between `min` and `max`, evalaute equation and compute output.
All input-output pairs are saved in `fname`.

# Arguments
  - `fmuPath::String`:                Path to FMU.
  - `fname::String`:                  File name to save training data to.
  - `eqId::Int64`:                    Index of equation to generate training data for.
  - `inputVars::Array{String}`:       Array with names of input variables.
  - `min::AbstractVector{<:Number}`:  Array with minimum value for each input variable.
  - `max::AbstractVector{<:Number}`:  Array with maximum value for each input variable.
  - `outputVars::Array{String}`:      Array with names of output variables.

# Keywords
  - `N::Integer = 1000`: Number of input-output pairs to generate.

See also [`generateFMU`](@ref), [`generateFMU`](@ref).
"""
function generateTrainingData(fmuPath::String, fname::String, eqId::Int64, inputVars::Array{String}, min::AbstractVector{<:Number}, max::AbstractVector{<:Number}, outputVars::Array{String}; N::Integer=1000)
  #ENV["JULIA_DEBUG"] = "FMICore"

  # Handle time inut variable
  usesTime = false
  local timeValues
  loc = findall(elem->elem=="time", inputVars)
  if length(loc) >= 1
    usesTime = true
    loc = first(loc)
    timeValues = sort((max[loc]-min[loc]).*rand(N) .+ min[loc])
    deleteat!(inputVars, loc)
    deleteat!(min, loc)
    deleteat!(max, loc)
  end

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs+nOutputs

  @assert length(min) == length(max) == nInputs "Length of min, max and inputVars doesn't match"

  # Create empty data frame
  col_names = Symbol.(vcat(inputVars, outputVars))
  col_types = fill(Float64, nVars)
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types )
  df = DataFrames.DataFrame(named_tuple)

  local fmu
  Suppressor.@suppress begin
    fmu = FMI.fmiLoad(fmuPath)
  end
  try
    # Load FMU and initialize
    FMI.fmiInstantiate!(fmu; loggingOn = false, externalCallbacks=false)

    FMI.fmiSetupExperiment(fmu, 0.0, 1.0)
    FMI.fmiEnterInitializationMode(fmu)
    FMI.fmiExitInitializationMode(fmu)

    # Generate training data
    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars,outputVars))

    ProgressMeter.@showprogress 1 "Generating training data ..." for i in 1:N
      # Set input values with random values
      row[1:nInputs] = (max.-min).*rand(nInputs) .+ min
      # Set start values to 0?
      row[nInputs+1:end] .= 0.0
      FMIImport.fmi2SetReal(fmu, row_vr, row)
      if usesTime
        FMIImport.fmi2SetTime(fmu, timeValues[i])
      end

      # Evaluate
      status = fmiEvaluateEq(fmu, eqId)
      if status != fmi2OK
        continue
      end

      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])

      # Update data frame
      push!(df, row)
    end
  catch err
    rethrow(err)
  finally
    FMI.fmiUnload(fmu)
  end
  mkpath(dirname(fname))
  CSV.write(fname, df)
  return fname
end
