using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using BSON
using DataFrames
using Flux
import CSV
import FMI
import FMICore
import FMIImport
import InvertedIndices
import ONNXNaiveNASflux

include("/mnt/home/ph/ws/NonLinearSystemNeuralNetworkFMU.jl/src/fmiExtensions.jl")

# TODO hier muss noch richtig viel gemacht werden
# 1. der grundalgorithmus zum AL, also [samples erzeugen <-> netz trainieren]
# 2. die optimierungsalgorithmen (falls mehrere) z.B. bee-algo

abstract type OptMethod end

struct BeesOptMethod <: OptMethod
  "Step size of random walk."
  delta::Float64
  "Number of bees following best bees"
  nBest::Integer
  BeesOptMethod(; delta=1e-3) = new(delta)
end

struct ActiveLearnOptions
  "Method to generate data points"
  optmethod::OptMethod
  "Number of points to generate"
  samples::Integer
  "Number of active learning iterations"
  steps::Integer
  "Append to already existing data"
  append::Bool
  "Clean up temp CSV files"
  clean::Bool

  # Constructor
  function ActiveLearnOptions(; optmethod=BeesOptMethod(delta=1e-3)::OptMethod,
    samples=1000::Integer,
    steps=10::Integer,
    append=false::Bool,
    clean=true::Bool)
    new(optmethod, samples, steps, append, clean)
  end
end


function activeLearnWrapper(
  fmuPath::String,
  initialData::String,
  model,
  onnxFile::String,
  csvFile::String,
  eqId::Int64,
  inputVars::Array{String},
  minBound::Array{Float64},
  maxBound::Array{Float64},
  outputVars::Array{String};
  options=ActiveLearnOptions()::ActiveLearnOptions
)

  # Handle time input variable
  inputVarsCopy = copy(inputVars)
  usesTime = false
  timeBounds = nothing
  loc = findall(elem -> elem == "time", inputVars)
  if length(loc) >= 1
    @assert length(loc) == 1 "time variable occurs more than once"
    usesTime = true
    loc = first(loc)
    timeBounds = (minBound[loc], maxBound[loc])
    deleteat!(inputVarsCopy, loc)
    deleteat!(minBound, loc)
    deleteat!(maxBound, loc)
  end

  fmu = FMI.fmiLoad(fmuPath)
  activeLearn(fmu, initialData, model, onnxFile, csvFile, eqId, timeBounds, inputVarsCopy, minBound, maxBound, outputVars; options=options)
  FMI.fmiUnload(fmu)
end


function activeLearn(
  fmu,
  initialData::String,
  model,
  onnxFile::String,
  csvFile::String,
  eqId::Int64,
  timeBounds::Union{Tuple{Number,Number},Nothing},
  inputVars::Array{String},
  inMin::Array{Float64},
  inMax::Array{Float64},
  outputVars::Array{String};
  options::ActiveLearnOptions
)

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs + nOutputs
  useTime = timeBounds !== nothing

  samplesGenerated = 0

  @assert length(inMin) == length(inMax) == nInputs "Length of min, max and inputVars doesn't match"

  try
    # Load FMU and initialize
    FMI.fmiInstantiate!(fmu; loggingOn=false, externalCallbacks=false)

    if useTime
      FMI.fmiSetupExperiment(fmu, timeBounds[1], timeBounds[2])
    else
      FMI.fmiSetupExperiment(fmu)
    end
    FMI.fmiEnterInitializationMode(fmu)
    FMI.fmiExitInitializationMode(fmu)

    # Get initial training data
    df = DataFrames.select(CSV.read(initialData, DataFrames.DataFrame; ntasks=1), vcat(inputVars, outputVars))
    @info "\n$(names(df))\n$(vcat(inputVars,outputVars))"
    data = prepareData(df, inputVars, outputVars)

    # Initial training run
    model, df_loss = trainSurrogate!(model, data.train, data.test)

    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars, outputVars))

    for step in 1:options.steps
      @info "Step $(step):"

      # Set input values with random values
      row[1:nInputs] = (inMax .- inMin) .* rand(nInputs) .+ inMin
      # Set Output values with surrogate
      row[nInputs+1:end] .= model(row[1:nInputs])

      status, res = fmiEvaluateRes(fmu, eqId, row[nInputs+1:end])
      @info "res $res"

      if useTime
        status, row = generateDataPoint(fmu, eqId, nInputs, row_vr, row, timeBounds[1])
      else
        status, row = generateDataPoint(fmu, eqId, nInputs, row_vr, row, nothing)
      end

      model, df_loss = trainSurrogate!(model, data.train, data.test)
      #@info "$df_loss"

      # Found a point: stop
      if status == fmi2OK
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

    # save model and data
    mkpath(dirname(onnxFile))
    BSON.@save onnxFile * ".bson" model
    ONNXNaiveNASflux.save(onnxFile, model, (nInputs, 1))

    mkpath(dirname(csvFile))
    CSV.write(csvFile, df)

  catch err
    FMI.fmiUnload(fmu)
    rethrow(err)
  end
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
