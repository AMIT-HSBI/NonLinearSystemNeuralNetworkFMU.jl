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
    generateDataBatch(fmu, fname, eqId, timeBounds, inputVars, inMin, inMax, outputVars, p;
                      n, options) where T <: Number

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
  - `samples::Integer`:                       Number of input-output pairs to generate.
  - `options::DataGenOptions:                 Data generation settings.
"""
function generateDataBatch(fmu,
                           fname::String,
                           eqId::Int64,
                           timeBounds::Union{Tuple{T,T}, Nothing},
                           inputVars::Array{String},
                           inMin::AbstractVector{T},
                           inMax::AbstractVector{T},
                           outputVars::Array{String},
                           p::ProgressMeter.Progress;
                           samples::Integer,
                           options::DataGenOptions) where T <: Number

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs+nOutputs
  useTime = timeBounds !== nothing

  samplesGenerated = 0

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
    nFailures = 0
    while !found && nFailures<10
      # Set input values with random values
      row[1:nInputs] = (inMax.-inMin).*rand(nInputs) .+ inMin
      # Set start values to 0?
      # TODO start values from Modelica attributes?
      row[nInputs+1:end] .= 0.0

      status, row = generateDataPoint(fmu, eqId, nInputs, row_vr, row, if useTime timeBounds[1] else nothing end)

      # Found a point: stop
      if status == fmi2OK
        ProgressMeter.next!(p)
        found = true
        samplesGenerated += 1

        # Update data frame
        if useTime
          push!(df, vcat([timeBounds[1]], row))
        else
          push!(df, row)
        end
      else
        nFailures += 1
      end
    end

    if !found
      @warn "No initial solution found"
    else
      timeValues = nothing
      if useTime
        # TODO time always increases, is this necessary
        timeValues = sort((timeBounds[2]-timeBounds[1]).*rand(N-1) .+ timeBounds[1])
      end

      # Generate remaining points
      # give up after too many consecutive fails
      nFailures = 0
      while samplesGenerated < samples && nFailures < 10
        if typeof(options.method) === RandomMethod
          row[1:nInputs] = (inMax.-inMin).*rand(nInputs) .+ inMin
        elseif typeof(options.method) == RandomWalkMethod
          randomStep!(view(row,1:nInputs), inMin, inMax, delta=options.method.delta)
        else
          error("Unknown method '$(typeof(options.method))'");
        end

        status, row = generateDataPoint(fmu, eqId, nInputs, row_vr, row, if useTime timeValues[i] else nothing end)

        if status == fmi2OK
          ProgressMeter.next!(p)
          nFailures = 0
          samplesGenerated += 1

          # Update data frame
          if useTime
            push!(df, vcat([timeValues[i]], row))
          else
            push!(df, row)
          end
        else
          # Reset start value of iteration
          row[nInputs+1:end] .= 0.0
          nFailures += 1
          continue
        end
      end

      if nFailures >= 10
        @warn "No solution found after $samplesGenerated samples"
      end
    end
  catch err
    FMI.fmiUnload(fmu)
    rethrow(err)
  end

  mkpath(dirname(fname))
  CSV.write(fname, df)
end


"""
    generateTrainingData(fmuPath, workDir, fname, eqId, inputVars, min, max, outputVars;
                         options=DataGenOptions())

Generate training data for given equation of FMU.

Generate random inputs between `min` and `max`, evalaute equation and compute output.
All input-output pairs are saved in CSV file `fname`.

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
  - `options::DataGenOptions`:        Settings for data generation.

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
                              options=DataGenOptions()::DataGenOptions) where T <: Number

  # Handle time input variable
  inputVarsCopy = copy(inputVars)
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

  # Generate data batch wise
  @info "Starting data generation on $(options.nBatches) batches with $(options.nThreads) threads."
  progressMeter = ProgressMeter.Progress(options.n; desc="Generating training data ...")
  nPerBatch = Integer(ceil(options.n / options.nBatches))
  batchesDone = 0
  while batchesDone < options.nBatches
    parallelBatches = min(options.nThreads, options.nBatches-batchesDone)
    # A FMU that is instantiate once can't be reused by another (or the same) thread.
    fmuArray = [FMI.fmiLoad(fmuPath) for _ in 1:parallelBatches]

    Threads.@threads for i in 1:parallelBatches  # enumerate not thread safe
      fmu = fmuArray[i]
      @debug "Thread $(Threads.threadid()) running FMU $i"
      tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_batch_$(batchesDone+i).csv")
      samples = nPerBatch
      if i == parallelBatches
        samples = options.n - nPerBatch*(options.nBatches-1)
      end
      generateDataBatch(fmu, tempCsvFile, eqId, timeBounds, inputVarsCopy, minBound, maxBound, outputVars, progressMeter; samples, options=options)
    end

    FMI.fmiUnload.(fmuArray)
    batchesDone += parallelBatches
  end
  ProgressMeter.finish!(progressMeter)

  # Combine CSV files from all braches
  mkpath(dirname(fname))
  for i = 1:options.nBatches
    tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_batch_$(i).csv")
    df = CSV.read(tempCsvFile, DataFrames.DataFrame; ntasks=1)
    df[!, "Trace"] .= i
    if i==1
      CSV.write(fname, df; append=options.append)
    else
      CSV.write(fname, df; append=true)
    end
    if options.clean
      rm(tempCsvFile)
    end
  end

  return fname
end

"""
Evaluate equation `eqId` with `row` as inputs + start values
"""
function generateDataPoint(fmu, eqId, nInputs, row_vr, row, time)
  # Set input values and start values for output
  FMIImport.fmi2SetReal(fmu, row_vr, row)
  if time !== nothing
    FMIImport.fmi2SetTime(fmu, time)
  end

  # Evaluate equation
  # TODO: Supress stream prints, but Suppressor.jl is not thread safe
  status = fmiEvaluateEq(fmu, eqId)
  if status == fmi2OK
    # Get output values
    row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])
  end

  return status, row
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
