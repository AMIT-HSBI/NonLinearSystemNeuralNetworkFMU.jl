include("convertProximityData.jl")

using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using BSON
using Flux

"""
Train Flux model that uses previous solution as additional input.

y_i = model(x_i, y_i-1)
"""
function trainFlux(modelName, N; nepochs=100, losstol=1e-8)
  workdir = datadir("sims", "$(modelName)_$(N)")
  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])[1:1]

  # Convert data CSV to proximity data CSV
  (csvFile_proximity, df_proximity) = convertProximityData(workdir, N, profilingInfo[1].eqInfo.id)

  # Train ONNX
  onnxFiles = String[]
  nInputs = length(profilingInfo[1].usingVars) + length(profilingInfo[1].iterationVariables)
  nOutputs = length(profilingInfo[1].iterationVariables)
  for (i, prof) in enumerate(profilingInfo)
    model = Flux.Chain(Flux.Dense(nInputs,     nInputs*20,  Flux.σ),
                       Flux.Dense(nInputs*20,  nOutputs*10, tanh),
                       Flux.Dense(nOutputs*10, nOutputs*10, tanh),
                       Flux.Dense(nOutputs*10, nOutputs))
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
  fmu_interface = joinpath(workdir, modelName*".interface.fmu")
  tempDir = joinpath(workdir, "temp")
  fmu_onnx = buildWithOnnx(fmu_interface,
                          modelName,
                          profilingInfo,
                          onnxFiles;
                          tempDir=workdir,
                          usePrevSol=true)
  rm(tempDir, force=true, recursive=true)
end
