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
  - `N::Integer = 1000`:   Number of input-output pairs to generate.
  - `delta::T = 1e-3`:     Stepsize of random walk relative to the input space size
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
                     delta::T = 1e-3) where T <: Number

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
    for _ in 1:10
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

      # Found a point: break
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
        break
      end
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
        # Move input values in random direction
        row[1:nInputs] .+= (inMax.-inMin).*(2.0 .*rand(nInputs) .- 1.0).*delta
        # Check bounds
        row[1:nInputs] .= max.(row[1:nInputs], inMin)
        row[1:nInputs] .= min.(row[1:nInputs], inMax)
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
  - `min::AbstractVector{T}`:         Array with minimum value for each input variable.
  - `max::AbstractVector{T}`:         Array with maximum value for each input variable.
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
  timeBounds = nothing
  loc = findall(elem->elem=="time", inputVars)
  if length(loc) >= 1
    @assert length(loc) == 1 "time variable occurs more than once"
    usesTime = true
    loc = first(loc)
    timeBounds = (min[loc], max[loc])
    deleteat!(inputVarsCopy, loc)
    deleteat!(min, loc)
    deleteat!(max, loc)
  end

  @info "Starting data generation on $(nBatches) batches."
  p = ProgressMeter.Progress(N_perThread*nBatches; desc="Generating training data ...")
  local fmuBatch
  Suppressor.@suppress begin
    fmuBatch = [(i, FMI.fmiLoad(fmuPath)) for i in 1:nBatches]
  end
  Threads.@threads for (i, fmu) in fmuBatch
    tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_tread_$(i).csv")
    simulateFMU(fmu, tempCsvFile, eqId, timeBounds, inputVarsCopy, min, max, outputVars, p; N=N_perThread)
  end
  ProgressMeter.finish!(p)

  # Combine CSV files
  mkpath(dirname(fname))
  for i = 1:nBatches
    tempCsvFile = joinpath(workDir, "trainingData_eq_$(eqId)_tread_$(i).csv")
    df = CSV.read(tempCsvFile, DataFrames.DataFrame; ntasks=1)
    if i==1
      CSV.write(fname, df; append=append)
    else
      CSV.write(fname, df; append=true)
    end
    rm(tempCsvFile);
  end

  return fname
end
