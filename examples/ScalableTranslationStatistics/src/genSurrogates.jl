using NaiveONNX
using NonLinearSystemNeuralNetworkFMU
using Flux

include(srcdir("util.jl"))

function genSurrogate(lib::String, modelName::String; n::Int=1000, genData::Bool=true)
  # Get lib and model
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  workdir = datadir("sims", split(modelName, ".")[end])

  # Generate training data
  local csvFiles
  local fmu
  local profilingInfo
  if genData
    omOptions = OMOptions(workingDir=workdir)
    dataGenOptions = DataGenOptions(method=RandomWalkMethod(delta=0.01),
                                    n = n,
                                    nBatches = 100,
                                    nThreads = max(1, Integer(floor(Threads.nthreads() * 0.9))))
    (csvFiles, fmu, profilingInfo) = main(modelName, [lib], omOptions=omOptions, dataGenOptions=dataGenOptions, reuseArtifacts=false)
  else
    profilingInfo = getProfilingInfo(joinpath(workdir, "profilingInfo.bson"))
    csvFiles = [joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv") for prof in profilingInfo]
    fmu = joinpath(workdir, "$(modelName).fmu")
  end

  # Train ONNX
  onnxFiles = String[]
  for (i ,prof) in enumerate(profilingInfo)
    onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
    lossFile = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id)_loss.csv"))
    push!(onnxFiles, onnxModel)

    #nInputs = length(prof.usingVars)
    #nOutputs = length(prof.iterationVariables)  
    #model = Flux.Chain(Flux.Dense(nInputs,   nInputs*5, Flux.sigmoid),
    #                   Flux.Dense(nInputs*5, nOutputs))
    model = nothing
    @showtime trainONNX(csvFiles[i], onnxModel, prof.usingVars, prof.iterationVariables; lossFile=lossFile, model=model, nepochs=1000, losstol=1e-6)
  end

  # Include ONNX into FMU
  fmu_interface = joinpath(workdir, modelName*".interface.fmu")
  tempDir = joinpath(workdir, "temp-onnx")
  @showtime fmu_onnx = buildWithOnnx(fmu_interface,
                                     modelName,
                                     profilingInfo,
                                     onnxFiles;
                                     tempDir=tempDir)

  # Save FMU artifacts
  fmuDir = datadir("sims", split(modelName, ".")[end], "fmus")
  if !isdir(fmuDir)
    mkpath(fmuDir)
  end
  cp(fmu, joinpath(fmuDir, modelName*".fmu"), force=true)
  cp(fmu_onnx, joinpath(fmuDir, modelName*".onnx.fmu"), force=true)

  return (profilingInfo, fmu, fmu_onnx)
end

function logProfilingInfo(sizes, logFile)
  file = open(logFile, "w")
  for size in sizes
    (shortName, modelName) = getNames(size)
    write(file, "$(modelName)\n")
    profilingInfo = getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))

    for prof in profilingInfo
      write(file, dumpLoopInfo(prof, indentation=1))
    end
  end
  close(file)
end

function dumpLoopInfo(profilingInfo::ProfilingInfo; indentation=0)
  indent = repeat('\t', indentation)
  """
  $(indent)Equation $(profilingInfo.eqInfo.id)
  $(indent)\tNumber of iteration variables: $(length(profilingInfo.iterationVariables))
  $(indent)\tNumber of inner variables: $(length(profilingInfo.innerEquations))
  $(indent)\tNumber of used variables: $(length(profilingInfo.usingVars))
  $(indent)\tTotal evaluation time: $(profilingInfo.eqInfo.time) [s]
  """
end
