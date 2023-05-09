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
    simulateFMU(fmu, fname, eqId, timeBounds, inputVars, inMin, inMax, outputVars, p;
                N = 1000, delta = 1e-3) where T <: Number

Generate data points for given equation of FMU.

Generate random inputs between `inMin` and `inMax`, evalaute equation and compute output.
All input-output pairs are saved in `fname`.

# Arguments
  - `fmu`:                                    Instance of the FMU struct.
  - `fname::String`:                          File name to save training data to.
  - `eqId::Int64`:                            Index of equation to generate training data for.
  - `timeBounds::Union{Tuple{T,T}, Nothing}`: Minimum and maximum for time if it is an input variable, otherwise `nothing`.
  - `inputVars::Array{String}`:               Array with names of input variables.
  - `inMin::AbstractVector{T}`:               Array with minimum value for each input variable.
  - `inMax::AbstractVector{T}`:               Array with maximum value for each input variable.
  - `outputVars::Array{String}`:              Array with names of output variables.
  - `p::ProgressMeter.Progress`:              ProgressMeter to show computation progress.

# Keywords
  - `N::Integer = 1000`:                      Number of input-output pairs to generate.
  - `method::Symbol = :randomWalk`:           Available methods are: `:random` and `:randomWalk`.
"""
function simulateFMU(fmu,
                     fname::String,
                     eqId::Int64,
                     timeBounds::Union{Tuple{T,T}, Nothing},
                     inputVars::Array{String},
                     inMin::AbstractVector{T},
                     inMax::AbstractVector{T},
                     outputVars::Array{String},
                     p::ProgressMeter.Progress;
                     N::Integer = 1000,
                     method::Symbol = :randomWalk) where T <: Number

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs+nOutputs
  useTime = timeBounds !== nothing

  @assert length(inMin) == length(inMax) == nInputs "Length of min, max and inputVars doesn't match"

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

    if useTime
      FMI.fmiSetupExperiment(fmu, timeBounds[1], timeBounds[2])
    else
      FMI.fmiSetupExperiment(fmu)
    end
    FMI.fmiEnterInitializationMode(fmu)
    FMI.fmiExitInitializationMode(fmu)

    # Generate training data
    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars,outputVars))

    # Generate first point (try a few times)
    found = false
    k = 0
    while !found && k<10
      # Set input values with random values
      row[1:nInputs] = (inMax.-inMin).*rand(nInputs) .+ inMin
      # Set start values to 0?
      # TODO start values from Modelica attributes?
      row[nInputs+1:end] .= 0.0
      FMIImport.fmi2SetReal(fmu, row_vr, row)
      if useTime
        FMIImport.fmi2SetTime(fmu, timeBounds[1])
      end

      # Evaluate equation
      # TODO: Suppress stream prints, but Suppressor.jl is not thread safe
      status = fmiEvaluateEq(fmu, eqId)

      # Found a point: stop
      if status == fmi2OK
        ProgressMeter.next!(p)
        found = true
        # Get output values
        row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])

        # Update data frame
        if useTime
          push!(df, vcat([timeBounds[1]], row))
        else
          push!(df, row)
        end
      end
      k += 1
    end

    if !found
      @warn "No initial solution found"
    else
      # Generate remaining points by doing a random walk
      timeValues = nothing
      if useTime
        # TODO time always increases, is this necessary
        timeValues = sort((timeBounds[2]-timeBounds[1]).*rand(N-1) .+ timeBounds[1])
      end
      for i in 1:N-1
        if method == :random
          row[1:nInputs] = (inMax.-inMin).*rand(nInputs) .+ inMin
        elseif method == :randomWalk
          randomStep!(view(row,1:nInputs), inMin, inMax)
        else
          error("Unknown method " * String(method));
        end

        # Keep start values from previous step
        #row[nInputs+1:end] .= 0.0
        FMIImport.fmi2SetReal(fmu, row_vr, row)
        if useTime
          FMIImport.fmi2SetTime(fmu, timeValues[i])
        end

        # Evaluate equation
        # TODO: Supress stream prints, but Suppressor.jl is not thread safe
        status = fmiEvaluateEq(fmu, eqId)
        if status != fmi2OK
          @warn "No solution found"
          continue
        end
        ProgressMeter.next!(p)
        # Get output values
        row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])

        # Update data frame
        if useTime
          push!(df, vcat([timeValues[i]], row))
        else
          push!(df, row)
        end
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
    generateTrainingData(fmuPath, workDir, fname, eqId, inputVars, min, max, outputVars;
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
  - `minBound::AbstractVector{T}`:    Array with minimum value for each input variable.
  - `maxBound::AbstractVector{T}`:    Array with maximum value for each input variable.
  - `outputVars::Array{String}`:      Array with names of output variables.

# Keywords
  - `N::Integer = 1000`:      Number of input-output pairs to generate.
  - `nBatches::Integer = 1`:  Number of batches to separate `N` into to generate data in parallel.
  - `append::Bool=false`:     Append to existing CSV file `fname` if true.

See also [`generateFMU`](@ref).
"""
function generateTrainingData(fmuPath::String,
                              workDir::String,
                              fname::String,
                              eqId::Int64,
                              inputVars::Array{String},
                              minBound::AbstractVector{T},
                              maxBound::AbstractVector{T},
                              outputVars::Array{String};
                              N::Integer = 1000,
                              nBatches::Integer = 1,
                              append::Bool=false) where T <: Number
  #ENV["JULIA_DEBUG"] = "FMICore"

  N_perBatch = Integer(ceil(N / nBatches))
  if N < nBatches
    nBatches = 1
  end

  inputVarsCopy = copy(inputVars)

  # Handle time input variable
  usesTime = false
  timeBounds = nothing
  loc = findall(elem->elem=="time", inputVars)
  if length(loc) >= 1
    @assert length(loc) == 1 "time variable occurs more than once"
    usesTime = true
    loc = first(loc)
    timeBounds = (minBound[loc], maxBound[loc])
    deleteat!(inputVarsCopy, loc)
    deleteat!(minBound, loc)
    deleteat!(maxBound, loc)
  end

  @info "Starting data generation on $(nBatches) batches."
  p = ProgressMeter.Progress(N*nBatches; desc="Generating training data ...")

  i = 0
  tmpnBatches = nBatches
  while tmpnBatches > 0
    @info "Mini-Batch $i"
    nMiniBatch = min(tmpnBatches, 4*Threads.nthreads())       # Make sure to not initialize too many FMUs at once
    tmpnBatches = tmpnBatches - nMiniBatch
    local fmuBatch
    Suppressor.@suppress begin
      fmuBatch = [(j, FMI.fmiLoad(fmuPath)) for j in 1:nMiniBatch]
    end
    Threads.@threads for (j, fmu) in fmuBatch
      idx = i + j
      tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_tread_$(idx).csv")
      simulateFMU(fmu, tempCsvFile, eqId, timeBounds, inputVarsCopy, minBound, maxBound, outputVars, p; N=N_perBatch)
    end
    i += nMiniBatch
  end
  ProgressMeter.finish!(p)

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
Move point in a random direction with step size delta*(boundaryMax.-boundaryMin)
while staying in boundary.
"""
function randomStep!(point::AbstractVector{T},
                     boundaryMin::AbstractVector{T},
                     boundaryMax::AbstractVector{T};
                     delta::Float64 = 1e-3) where T <: AbstractFloat
  point .+= (boundaryMax.-boundaryMin).*(2.0 .*rand(length(point)) .- 1.0) .* delta
  # Check boundaries
  point .= max.(point, boundaryMin)
  point .= min.(point, boundaryMax)
end

"""
Transform data set into proximity data set.

Take data point (x,y) and add y_tile to the input from a data point in close proximity.
Save new datapoint ([x,y_tile], y).
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
