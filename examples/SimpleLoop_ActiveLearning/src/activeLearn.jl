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
    data = prepareData(df_prox, vcat(inputVars, outputVars .* "_old"), outputVars)

    # Initial training run
    model, df_loss = trainSurrogate!(model, data.train, data.test; losstol=1e-3, nepochs=300)
    CSV.write("loss_0.csv", df_loss)

    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars, outputVars))


    function generateDataPoint(old_x; old_y=nothing, nCopies=8)
      x = Array{Float64}(undef, nInputs)
      y = Array{Float64}(undef, nOutputs)

      row = Array{Float64}(undef, nVars)
      # Set input values and start values for output
      row[1:nInputs] .= old_x
      if old_y === nothing
        # TODO get output from nearby input
        row[nInputs+1:end] .= 0.0
        #row[nInputs+1:end] .= model(row)
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
      x .= row[1:nInputs]
      y .= row[nInputs+1:end]
      # Update data frame
      if useTime
        push!(df, vcat([timeBounds[1]], x, y))
        for _ in 1:nCopies
          push!(df_prox, vcat([timeBounds[1]], x, wiggle.(y), y))
        end
      else
        push!(df, vcat(x, y))
        for _ in 1:nCopies
          push!(df_prox, vcat(x, wiggle.(y), y))
        end
      end

      samplesGenerated += 1

      return row[nInputs+1:end]
    end


    function makeRandInputOutput()
      x = Array{Float64}(undef, nInputs)
      y = Array{Float64}(undef, nOutputs)

      # Set input values with random values
      x = (inMax .- inMin) .* rand(nInputs) .+ inMin

      # Set initial guess to correct output
      # TODO get output from nearby input
      y .= generateDataPoint(x)

      # Set output values with surrogate
      y .= model(vcat(x, wiggle.(y)))

      status, res = fmiEvaluateRes(fmu, eqId, vcat(x, y))
      @assert status == fmi2OK "residual could not be evaluated"

      return vcat(x, y), sum(res .^ 2)
    end


    function makeNeighbor(old_row; delta=0.03)
      # Make random neighbor of old x
      x = old_row[1:nInputs] + (inMax .- inMin) .* (2.0 .* rand(nInputs) .- 1.0) .* delta
      # Check boundaries
      x .= max.(x, inMin)
      x .= min.(x, inMax)

      # Set initial guess to old y
      y = old_row[nInputs+1:end]

      # maybe
      y .= generateDataPoint(x; old_y=y)

      # Evaluate surrogate
      y .= model(vcat(x, y))

      # Evaluate residual
      status, res = fmiEvaluateRes(fmu, eqId, vcat(x, y))
      @assert status == fmi2OK "could not evaluated residual $eqId"

      return vcat(x, y), sum(res .^ 2)
    end

    col_names = Symbol.(vcat(inputVars, "res_norm"))
    col_types = fill(Float64, nInputs + 1)
    named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types)
    x = Array{Float64}(undef, nInputs)
    y = Array{Float64}(undef, nOutputs)
    function resLandscape(fname::String, start::Integer)
      # Create new empty data frame
      df_res = DataFrames.DataFrame(named_tuple)

      for row in eachrow(df)[start:end]
        x .= Vector(row[1:nInputs])
        y .= Vector(row[nInputs+1:end])
        y .= model(vcat(x, y))
        status, res = fmiEvaluateRes(fmu, eqId, vcat(x, y))
        @assert status == fmi2OK "residual could not be evaluated"
        res = sum(res .^ 2)
        push!(df_res, vcat(x, res))
      end

      CSV.write(fname, df_res)
    end

    resLandscape("res_0.csv", 1)

    for step in 1:options.steps
      @info "Step $(step):"

      samples = floor(Integer, (options.samples - samplesGenerated)/(options.steps-step+1))
      if samples <= 0
        break
      end

      start = length(eachrow(df)) + 1
      @info "start $start"
      beesAlgorithm(makeRandInputOutput, makeNeighbor; samples=samples)
      data = prepareData(df_prox, vcat(inputVars, outputVars .* "_old"), outputVars)

      resLandscape("res_$step.csv", start)

      # Train model with augmented data set
      model, df_loss = trainSurrogate!(model, data.train, data.test; losstol=1e-5, nepochs=300)
      CSV.write("loss_$step.csv", df_loss)
    end
    @info "generated $samplesGenerated"

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

    Find maximal value.

# Arguments
  - `new`:    function to generate a new random bee
  - `next`:   function to generate a bee in the neighborhood of bee (x,y)
  - `gen`:    function to actually generate the sample point (x, initialGuessY)

# Keywords
  - `samples::Integer`:   number of samples to generate during optimization

"""
function beesAlgorithm(new, next; samples::Integer)
  popsize = floor(Integer, samples*0.1)
  nBest = floor(Integer, popsize*0.25)
  nBestNeighbors = 5

  x = Array{Any}(undef,popsize)
  y = Array{Float64}(undef,popsize)

  # make starting population
  for i in 1:popsize
    x[i], y[i] = new()
  end
  generated = popsize

  # main loop
  while generated < samples
    p = sortperm(y,rev=true)
    for i in p[1:nBest]
      # best bees look at neighbors
      for _ in 1:nBestNeighbors
        if generated < samples
          xn, yn = next(x[i])
          generated += 1
        else
          break
        end
        # if new is better, replace
        if yn > y[i]
          x[i] .= xn
          y[i] = yn
        end
      end
    end
    if generated >= samples
      break
    end
    for i in p[nBest:end]
      # rest looks randomly
      if generated >= samples
        x[i], y[i] = new()
        generated += 1
      else
        break
      end
    end
  end
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
  nCopies=8::Integer
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
    for _ in 1:nCopies
      y = Vector(row[nInputs+1:nInputs+nOutputs])
      y_tilde = wiggle.(y)
      x = Vector(row[1:nInputs])
      new_row = vcat(x, y_tilde, y)
      push!(df_proximity, new_row)
    end
  end
  return df_proximity
end

function wiggle(x; delta=0.01)
  r = delta * (2 * rand() - 1)
  return x + (r * x)
end
