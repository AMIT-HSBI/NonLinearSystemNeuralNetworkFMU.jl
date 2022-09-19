# $julia -e "include(\"ScalableTranslationStatistics.jl\"); run()"
# $nohup julia-1.8.1 -e "include(\"ScalableTranslationStatistics.jl\"); run()" &

cd(@__DIR__)
using NaiveONNX
import Pkg; Pkg.activate("..")
using NonLinearSystemNeuralNetworkFMU
using BSON: @save, @load

ENV["ORT_DIR"] = "/mnt/home/aheuermann/workdir/julia/NonLinearSystemNeuralNetworkFMU.jl/onnxruntime-linux-x64-1.12.1"
rootDir = "/mnt/home/aheuermann/workdir/phymos/ScalableTranslationStatistics"

function runScalableTranslationStatistics(rootDir::String; level::Integer=1, N::Integer=10000)
  # Get lib and model
  lib = joinpath(rootDir, "02_SourceModel", "02_Model", "01_AuthoringModel", "ScalableTranslationStatistics", "package.mo")
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  size = ["5", "10", "20", "40", "80", "100"]
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size[level])"

  # Profile non-linear loops
  @showtime profilingInfo = profiling(modelName, lib; threshold=0)
  allProfiling = joinpath("profiling", "level$(string(level))")
  mkpath(allProfiling)
  @save joinpath(allProfiling, "profilingInfo.bson") profilingInfo

  # Find min-max values of used varaibles
  allUsedvars = vcat([prof.usingVars for prof in profilingInfo]...)
  @showtime (min, max) = minMaxValuesReSim(allUsedvars, modelName, lib)

  #plot(allUsedvars, min, xrotation=90, label="min", legend=:topleft)
  #plt = plot!(allUsedvars, max, label="max")
  #savefig(plt,"$(modelName)_minmax.svg")

  # Generate default FMU
  @showtime fmu = generateFMU(modelName, lib, workingDir = modelName)
  allFmuDir = joinpath("fmus", "level$(string(level))")
  mkpath(allFmuDir)
  cp(fmu, joinpath(allFmuDir, basename(fmu)), force=true)
  fmu = joinpath(allFmuDir, basename(fmu))

  # Generate extended FMU
  allEqs = [prof.eqInfo.id for prof in profilingInfo]
  @showtime fmu_interface = addEqInterface2FMU(modelName, fmu, allEqs, workingDir = modelName)
  basename(fmu_interface)
  cp(fmu_interface, joinpath(allFmuDir, basename(fmu_interface)), force=true)
  fmu_interface = joinpath(allFmuDir, basename(fmu_interface))

  # Generate training data
  csvFiles = String[]
  for prof in profilingInfo
    eqIndex = prof.eqInfo.id
    inputVars = prof.usingVars
    outputVars = prof.iterationVariables

    mi = Array{Float64}(undef, length(inputVars))
    ma = Array{Float64}(undef, length(inputVars))
  
    for (i,var) in enumerate(inputVars)
      idx = findfirst(x->x==var, allUsedvars)
      mi[i] = min[idx]
      ma[i] = max[idx]
    end

    fileName = abspath(joinpath("data", "level$(string(level))", "eq_$(prof.eqInfo.id).csv"))
    @showtime csvFile = generateTrainingData(fmu_interface, fileName, eqIndex, inputVars, mi, ma, outputVars; N = N)
    push!(csvFiles, csvFile)
  end

  # Train ONNX
  onnxFiles = String[]
  for (i ,prof) in enumerate(profilingInfo)
    onnxModel = abspath(joinpath("onnx", "level$(string(level))", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)
    nInputs = length(prof.usingVars)

    @showtime trainONNX(csvFiles[i], onnxModel, nInputs; nepochs=100, losstol=1e-4)
  end

  # Include ONNX into FMU
  @showtime fmu_onnx = buildWithOnnx(fmu_interface,
                                     modelName,
                                     profilingInfo,
                                     onnxFiles;
                                     tempDir=modelName)
  basename(fmu_onnx)
  cp(fmu_onnx, joinpath(allFmuDir, basename(fmu_onnx)), force=true)
  fmu_onnx = joinpath(allFmuDir, basename(fmu_onnx))

  return (profilingInfo, fmu, fmu_onnx)
end

function clean()
  rm("data/", recursive=true, force=true)
  rm("fmus/", recursive=true, force=true)
  rm("onnx/", recursive=true, force=true)
  rm("profiling/", recursive=true, force=true)
  foreach(rm, filter(endswith(".log"), readdir(@__DIR__,join=true)))
  for dir in readdir(@__DIR__)
    if startswith(dir, "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_")
      rm(dir, recursive=true)
    end
  end
end

# Create all the FMUs
function run()
  for level in 1:6
    @info "Starting level $level"
    redirect_stdio(stdout="level$(string(level)).log", stderr="level$(string(level)).log") do
      @time runScalableTranslationStatistics(rootDir, level = level)
    end
  end
end
