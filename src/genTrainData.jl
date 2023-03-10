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


function simulateFMU(fmu,
                     fname::String,
                     eqId::Int64,
                     timeValues::Union{AbstractVector{T}, Nothing},
                     inputVars::Array{String},
                     min::AbstractVector{T},
                     max::AbstractVector{T},
                     outputVars::Array{String},
                     p::ProgressMeter.Progress;
                     N::Integer = 1000) where T <: Number

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs+nOutputs
  useTime = timeValues !== nothing

  @assert length(min) == length(max) == nInputs "Length of min, max and inputVars doesn't match"

  # Create empty data frame
  local col_names
  local col_types
  if useTime
    col_names = Symbol.(vcat("time", inputVars, outputVars))
    col_types = fill(Float64, nVars+1)
  else
    col_names = Symbol.(vcat(inputVars, outputVars))
    col_types = fill(Float64, nVars)
  end
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types)
  df = DataFrames.DataFrame(named_tuple)

  try
    # Load FMU and initialize
    FMI.fmiInstantiate!(fmu; loggingOn = false, externalCallbacks=false)

    FMI.fmiSetupExperiment(fmu, 0.0, 1.0)
    FMI.fmiEnterInitializationMode(fmu)
    FMI.fmiExitInitializationMode(fmu)

    # Generate training data
    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars,outputVars))

    for i in 1:N
      ProgressMeter.next!(p)
      # Set input values with random values
      row[1:nInputs] = (max.-min).*rand(nInputs) .+ min
      # Set start values to 0?
      row[nInputs+1:end] .= 0.0
      FMIImport.fmi2SetReal(fmu, row_vr, row)
      if useTime
        FMIImport.fmi2SetTime(fmu, timeValues[i])
      end

      # Evaluate equation
      # TODO: Supress stream prints, but Suppressor.jl is not thread safe
      status = fmiEvaluateEq(fmu, eqId)
      if status != fmi2OK
        continue
      end

      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])

      # Update data frame
      if useTime
        push!(df, vcat([timeValues[i]], row))
      else
        push!(df, row)
      end
    end
  catch err
    rethrow(err)
  finally
    FMI.fmiUnload(fmu)
  end

  mkpath(dirname(fname))
  CSV.write(fname, df)
end


"""
    generateTrainingData(fmuPath, workDir, fname, eqId, inputVars, min max, outputVars;
                         N=1000, nBatches=1, append=false)

Generate training data for given equation of FMU.

Generate random inputs between `min` and `max`, evalaute equation and compute output.
All input-output pairs are saved in `fname`.

# Arguments
  - `fmuPath::String`:                Path to FMU.
  - `workDir::String`:                Working directory for generateTrainingData.
  - `fname::String`:                  File name to save training data to.
  - `eqId::Int64`:                    Index of equation to generate training data for.
  - `inputVars::Array{String}`:       Array with names of input variables.
  - `min::AbstractVector{T}`:         Array with minimum value for each input variable.
  - `max::AbstractVector{T}`:         Array with maximum value for each input variable.
  - `outputVars::Array{String}`:      Array with names of output variables.

# Keywords
  - `N::Integer = 1000`:      Number of input-output pairs to generate.
  - `nBatches::Integer = 1`:  Number of batches to separate `N` into  to generate data in parallel.
  - `append::Bool=false`:     Append to existing CSV file `fname` if true.

See also [`generateFMU`](@ref), [`generateFMU`](@ref).
"""
function generateTrainingData(fmuPath::String,
                              workDir::String,
                              fname::String,
                              eqId::Int64,
                              inputVars::Array{String},
                              min::AbstractVector{T},
                              max::AbstractVector{T},
                              outputVars::Array{String};
                              N::Integer = 1000,
                              nBatches::Integer = 1,
                              append::Bool=false) where T <: Number
  #ENV["JULIA_DEBUG"] = "FMICore"

  N_perThread = Integer(ceil(N / nBatches))

  inputVarsCopy = copy(inputVars)

  # Handle time input variable
  usesTime = false
  local allTimeValues
  loc = findall(elem->elem=="time", inputVars)
  if length(loc) >= 1
    usesTime = true
    loc = first(loc)
    allTimeValues = sort((max[loc]-min[loc]).*rand(N_perThread*nBatches) .+ min[loc])
    deleteat!(inputVarsCopy, loc)
    deleteat!(min, loc)
    deleteat!(max, loc)
  end

  @info "Starting data generation on $(nBatches) batches."
  p = ProgressMeter.Progress(N; desc="Generating training data ...")
  local fmuBatch
  Suppressor.@suppress begin
    fmuBatch = [(i, FMI.fmiLoad(fmuPath)) for i in 1:nBatches]
  end
  Threads.@threads for (i, fmu) in fmuBatch
    tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_tread_$(i).csv")
    timeValues = nothing
    if usesTime
      timeValues = allTimeValues[(i-1)*N_perThread+1:(i*N_perThread)]
    end
    simulateFMU(fmu, tempCsvFile, eqId, timeValues, inputVarsCopy, min, max, outputVars, p; N=N_perThread)
  end

  # Combine CSV files
  mkpath(dirname(fname))
  for i = 1:nBatches
    tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_tread_$(i).csv")
    df = CSV.read(tempCsvFile, DataFrames.DataFrame; ntasks=1)
    df[!, "Trace"] .= i
    if i==1
      CSV.write(fname, df; append=append)
    else
      CSV.write(fname, df; append=true)
    end
    rm(tempCsvFile);
  end

  return fname
end

"""


"""
function data2proximityData(df::DataFrames.DataFrame,
                            inputVars::Array{String},
                            outputVars::Array{String};
                            neighbors=1::Integer,
                            weight=0.0::Float64)::DataFrames.DataFrame

  @assert in("Trace", names(df)) "DataFrame df is missing column 'Trace'."
  @assert weight >= 0 "weight has to be non-negative."
  @assert neighbors >= 1 "neighbors has to be larger than 0."

  nInputs = length(inputVars)
  nOutputs = length(outputVars)

  # Create new empty data frame
  col_names = Symbol.(vcat(inputVars, outputVars.*"_old", outputVars))
  col_types = fill(Float64, nInputs+2*nOutputs)
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types)
  df_proximity = DataFrames.DataFrame(named_tuple)

  # Fill new data frame for each trace
  for trace in sort(unique(df.Trace))
    df_trace = DataFrames.select(filter(row-> row.Trace == trace, df), InvertedIndices.Not([:Trace]))

    len = size(df_trace)[1]
    for (j,row) in enumerate(eachrow(df_trace))
      k = vcat(j-neighbors:j-1,  j+1:+j+neighbors)
      k = filter(k_i -> k_i > 0 && k_i < len, k)

      usingVars = Vector(row[1:nInputs])
      oldIterationVars = Vector(df_trace[rand(k), nInputs+1:nInputs+nOutputs])
      # Wiggle oldIterationVars
      if weight > 0
        oldIterationVars = wiggle.(oldIterationVars; weight=weight)
      end
      iterationVars = Vector(row[nInputs+1:nInputs+nOutputs])

      new_row = vcat(usingVars, oldIterationVars, iterationVars)
      push!(df_proximity, new_row)
    end
  end
  return df_proximity
end

function wiggle(x; weight=0.1)
  r = weight*(2*rand()-1)
  return x + (r*x)
end
