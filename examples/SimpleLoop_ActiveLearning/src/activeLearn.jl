using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using BSON
using Flux

# TODO hier muss noch richtig viel gemacht werden
# 1. der grundalgorithmus zum AL, also [samples erzeugen <-> netz trainieren]
# 2. die optimierungsalgorithmen (falls mehrere) z.B. bee-algo

function activeLearn(
  fmuPath::String,
  csvFile::String,
  eqId::Int64,
  inputVars::Array{String},
  minBound::AbstractVector{T},
  maxBound::AbstractVector{T},
  outputVars::Array{String};
  options=DataGenOptions()::DataGenOptions
# TODO options for ALtrain (contains `samples`, `bee-algo?` ... what else?)

) where {T<:Number}

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
  generateDataBatch(fmu, csvFile, eqId, timeBounds, inputVarsCopy, minBound, maxBound, outputVars; options=options)
  FMI.fmiUnload(fmu)
end





function generateDataBatch(fmu,
  csvFile::String,
  eqId::Int64,
  timeBounds::Union{Tuple{T,T},Nothing},
  inputVars::Array{String},
  inMin::AbstractVector{T},
  inMax::AbstractVector{T},
  outputVars::Array{String};
  options::DataGenOptions) where {T<:Number}

  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs + nOutputs
  useTime = timeBounds !== nothing

  samplesGenerated = 0

  @assert length(inMin) == length(inMax) == nInputs "Length of min, max and inputVars doesn't match"

  # Create empty data frame
  local col_names
  local col_types
  if useTime
    col_names = Symbol.(vcat("time", inputVars, outputVars))
    col_types = fill(Float64, nVars + 1)
  else
    col_names = Symbol.(vcat(inputVars, outputVars))
    col_types = fill(Float64, nVars)
  end
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types)
  df = DataFrames.DataFrame(named_tuple)

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

    # Generate training data
    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars, outputVars))

    # Generate first point (try a few times)
    found = false
    nFailures = 0
    while !found && nFailures < 10
      # Set input values with random values
      row[1:nInputs] = (inMax .- inMin) .* rand(nInputs) .+ inMin
      # Set start values to 0?
      # TODO start values from Modelica attributes?
      row[nInputs+1:end] .= 0.0

      status, row = generateDataPoint(fmu, eqId, nInputs, row_vr, row, if useTime
        timeBounds[1]
      else
        nothing
      end)

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
    end
  catch err
    FMI.fmiUnload(fmu)
    rethrow(err)
  end

  mkpath(dirname(csvFile))
  CSV.write(csvFile, df)
end


function trainONNX(csvFile::String,
  onnxModel::String,
  inputNames::Array{String},
  outputNames::Array{String};
  model,
  losstol::Real=1e-6,
  nepochs=10)

  data = readData(csvFile, inputNames, outputNames)
  nInputs = length(inputNames)
  nOutputs = length(outputNames)

  model = trainSurrogate!(model, data.train; losstol=losstol, nepochs=nepochs, useGPU=true)

  mkpath(dirname(onnxModel))
  BSON.@save onnxModel * ".bson" model
  ONNXNaiveNASflux.save(onnxModel, model, (nInputs, 1))

  return model
end

"""
Train Flux model with active learning

y = model(x)

generate at most `N` data samples
"""
function trainFlux(modelName, N; nepochs=100, losstol=1e-8)
  workdir = datadir("sims", "$(modelName)_$(N)")
  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])[1:1]

  # Train ONNX
  onnxFiles = String[]
  nInputs = length(profilingInfo[1].usingVars) + length(profilingInfo[1].iterationVariables)
  nOutputs = length(profilingInfo[1].iterationVariables)
  for (i, prof) in enumerate(profilingInfo)
    model = Flux.Chain(Flux.Dense(nInputs, nInputs * 20, Flux.σ),
      Flux.Dense(nInputs * 20, nOutputs * 10, tanh),
      Flux.Dense(nOutputs * 10, nOutputs * 10, tanh),
      Flux.Dense(nOutputs * 10, nOutputs)
    )
    onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)

    @showtime trainONNX(csvFile_proximity, onnxModel,
      names(df_proximity)[1:nInputs],
      names(df_proximity)[nInputs+1:nInputs+nOutputs];
      nepochs=nepochs,
      losstol=losstol,
      model=model)
  end

  # Include ONNX into FMU
  fmu_interface = joinpath(workdir, modelName * ".interface.fmu")
  tempDir = joinpath(workdir, "temp")
  fmu_onnx = buildWithOnnx(fmu_interface,
    modelName,
    profilingInfo,
    onnxFiles;
    tempDir=workdir,
    usePrevSol=true)
  rm(tempDir, force=true, recursive=true)
end
