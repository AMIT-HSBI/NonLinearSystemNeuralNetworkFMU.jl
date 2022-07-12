
include("fmiExtensions.jl")

"""
    generateTrainingData(fmuPath, fname, eqId, inputVars, min, max, outputVars[; N])

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
"""
function generateTrainingData(fmuPath::String, fname::String, eqId::Int64, inputVars::Array{String}, min::AbstractVector{<:Number}, max::AbstractVector{<:Number}, outputVars::Array{String}; N::Integer=1000)
  nInputs = length(inputVars)
  nOutputs = length(outputVars)
  nVars = nInputs+nOutputs

  # Create empty data frame
  col_names = Symbol.(vcat(inputVars, outputVars))
  col_types = fill(Float64, nVars)
  named_tuple = NamedTuple{Tuple(col_names)}(type[] for type in col_types )
  df = DataFrames.DataFrame(named_tuple)

  fmu = FMI.fmiLoad(fmuPath)
  try
    # Load FMU and initialize
    FMI.fmiInstantiate!(fmu, loggingOn = false)

    FMI.fmiSetupExperiment(fmu, 0.0, 1.0)
    FMI.fmiEnterInitializationMode(fmu)
    FMI.fmiExitInitializationMode(fmu)

    # Generate training data
    row = Array{Float64}(undef, nVars)
    row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, vcat(inputVars,outputVars))

    ProgressMeter.@showprogress 1 "Generating training data ..." for _ in 1:N
      # Set input values with random values
      row[1:nInputs] = (max.-min).*rand(nInputs) .+ min
      # Set start values to 0?
      row[nInputs+1:end] .= 0.0
      FMIImport.fmi2SetReal(fmu, row_vr, row)

      # Evaluate
      fmiEvaluateEq(fmu, eqId)

      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])

      # Update data frame
      push!(df, row)
    end
  catch err
    FMI.fmiUnload(fmu)
    rethrow(err)
  end
  mkpath(dirname(fname))
  CSV.write(fname, df)
end
