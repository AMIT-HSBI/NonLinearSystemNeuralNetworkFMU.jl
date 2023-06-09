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
  inMin::Array{Float64},
  inMax::Array{Float64},
  outputVars::Array{String};
  options=ActiveLearnOptions()::ActiveLearnOptions
)

  @assert length(inMin) == length(inMax) == length(inputVars) "Length of min, max and inputVars doesn't match"

  # Handle time input variable
  inputVarsCopy = copy(inputVars)
  usesTime = false
  timeBounds = nothing
  loc = findall(elem -> elem == "time", inputVars)
  if length(loc) >= 1
    @assert length(loc) == 1 "time variable occurs more than once"
    usesTime = true
    loc = first(loc)
    timeBounds = (inMin[loc], inMax[loc])
    deleteat!(inputVarsCopy, loc)
    deleteat!(inMin, loc)
    deleteat!(inMax, loc)
  end

  fmu = FMI.fmiLoad(fmuPath)
  activeLearn(fmu, initialData, model, onnxFile, csvFile, eqId, timeBounds, inputVarsCopy, inMin, inMax, outputVars; options=options)
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
    df = CSV.read(initialData, DataFrames.DataFrame; ntasks=1)
    df = DataFrames.select(df, vcat(inputVars, outputVars))
    df_prox = data2proximityData(df, inputVars, outputVars)
    @info "\n$(names(df))\n$(vcat(inputVars,outputVars))"
    data = prepareData(df_prox, vcat(inputVars, outputVars .* "_old"), outputVars)

    # Initial training run
    model, _ = trainSurrogate!(model, data.train, data.test)

    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars, outputVars))


    """
    """
    function generateDataPoint(old_x; old_y=nothing)
      row = Array{Float64}(undef, nVars)
      # Set input values and start values for output
      row[1:nInputs] .= old_x
      if old_y === nothing
        # TODO get output from nearby input
        row[nInputs+1:end] .= 0.0
        row[nInputs+1:end] .= model(row)
      else
        row[nInputs+1:end] .= old_y
      end
      FMIImport.fmi2SetReal(fmu, row_vr, row)
      if useTime
        FMIImport.fmi2SetTime(fmu, timeBounds[1])
      end

      # Evaluate equation
      # TODO: Supress stream prints, but Suppressor.jl is not thread safe
      status = fmiEvaluateEq(fmu, eqId)
      @assert status == fmi2OK "could not evaluate equation $eqId"

      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])
      # Update data frame
      if useTime
        new_row = vcat([timeBounds[1]], row[1:nInputs], wiggle.(row[nInputs+1:end]), row[nInputs+1:end])
      else
        new_row = vcat(row[1:nInputs], wiggle.(row[nInputs+1:end]), row[nInputs+1:end])
      end
      push!(df_prox, new_row)
      samplesGenerated += 1

      return row[nInputs+1:end]
    end


    """
    """
    function makeRandInputOutput()
      x = Array{Float64}(undef, nInputs)
      y = Array{Float64}(undef, nOutputs)

      # Set input values with random values
      x = (inMax .- inMin) .* rand(nInputs) .+ inMin

      # Set initial guess to correct output
      # TODO get output from nearby input
      y .= generateDataPoint(x)

      # Set output values with surrogate
      y .= model(vcat(x, y))

      status, res = fmiEvaluateRes(fmu, eqId, vcat(x, y))
      @assert status == fmi2OK "residual could not be evaluated"

      return vcat(x, y), sum(res .^ 2)
    end


    """
    """
    function makeNeighbor(old_row; delta=0.01)
      # Make random neighbor of old x
      x = old_row[1:nInputs] + (inMax .- inMin) .* (2.0 .* rand(nInputs) .- 1.0) .* delta
      # Check boundaries
      x .= max.(x, inMin)
      x .= min.(x, inMax)

      # Set initial guess to old y
      y = old_row[nInputs+1:end]

      # Evaluate surrogate
      y .= model(vcat(x, y))

      # Evaluate residual
      status, res = fmiEvaluateRes(fmu, eqId, vcat(x, y))
      @assert status == fmi2OK "could not evaluated residual $eqId"

      # maybe
      y .= generateDataPoint(x; old_y=y)

      return vcat(x, y), sum(res .^ 2)
    end


    for step in 1:options.steps
      @info "Step $(step):"

      generated = beesAlgorithm(makeRandInputOutput, makeNeighbor, generateDataPoint; samples=floor(Integer, (options.samples - samplesGenerated)/(options.steps-step+1)))
      if generated == 0
        break
      end

      # Train model with augmented data set
      data = prepareData(df_prox, vcat(inputVars, outputVars .* "_old"), outputVars)
      model, _ = trainSurrogate!(model, data.train, data.test)
    end

    # save model and data
    mkpath(dirname(onnxFile))
    BSON.@save onnxFile * ".bson" model
    ONNXNaiveNASflux.save(onnxFile, model, (nInputs + nOutputs, 1))

    mkpath(dirname(csvFile))
    CSV.write(csvFile, df)

  catch err
    FMI.fmiUnload(fmu)
    rethrow(err)
  end
end

"""
    beesAlgorithm(new, next, gen; samples)

# Arguments
  - `new`: function to generate a new random bee
  - `next`: function to generate a bee in the neighborhood of bee (x,y)
  - `gen`:

# Keywords
  - `samples::Integer`: number of samples to generate during optimization

"""
function beesAlgorithm(new, next, gen; samples::Integer)
  @info "samples $samples"
  generated = 0

  popsize = floor(Integer, samples/10)
  bestBees = floor(Integer, 0.5*popsize)

  optimum = (nothing, -Inf)

  x = Array{Any}(undef,samples)
  y = Array{Float64}(undef,samples)

  # starting population
  for i in 1:popsize
    x[i], y[i] = new()
    generated += 1
    @info "$i = $((x[i],y[i]))"
    if y[i] > optimum[2]
      optimum = (x[i], y[i])
    end
  end
  @info "optimum = $optimum"

  for _ in generated:samples-1
    x_new, y_new = next(optimum[1])
  end

  return generated
end


"""
Transform data set into proximity data set.

Take data point (x,y) and add y_tile = y + dy to the input.
Save new datapoint ([x,y_tile], y).
"""
function data2proximityData(
  df::DataFrames.DataFrame,
  inputVars::Array{String},
  outputVars::Array{String};
  delta=0.1::Float64,
  nCopies=4::Integer
)::DataFrames.DataFrame

  nInputs = length(inputVars)
  nOutputs = length(outputVars)

  # Create new empty data frame
  col_names = Symbol.(vcat(inputVars, outputVars .* "_old", outputVars))
  col_types = fill(Float64, nInputs + 2 * nOutputs)
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types)
  df_proximity = DataFrames.DataFrame(named_tuple)

  # Fill new data frame
  for row in eachrow(df)
    # Add copies with different y_tilde
    for i in 0:nCopies-1
      y = Vector(row[nInputs+1:nInputs+nOutputs])
      y_tilde = wiggle.(y; delta=delta * 0.95^i)
      x = Vector(row[1:nInputs])
      new_row = vcat(x, y_tilde, y)
      push!(df_proximity, new_row)
    end
  end
  return df_proximity
end

function wiggle(x; delta=0.1)
  r = delta * (2 * rand() - 1)
  return x + (r * x)
end
