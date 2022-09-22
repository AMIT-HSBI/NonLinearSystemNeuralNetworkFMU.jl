# $julia -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()"
# $nohup julia-1.8.1 -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()" &

cd(@__DIR__)
using NaiveONNX
import Pkg; Pkg.activate("../..")
using NonLinearSystemNeuralNetworkFMU
using BSON: @save, @load
using FMI
using FMICore
using CSV
using DataFrames
using BenchmarkTools
using Plots

ENV["ORT_DIR"] = "/mnt/home/aheuermann/workdir/julia/NonLinearSystemNeuralNetworkFMU.jl/onnxruntime-linux-x64-1.12.1"
rootDir = "/mnt/home/aheuermann/workdir/phymos/ScalableTranslationStatistics"

sizes = ["5", "10", "20", "40", "80", "160"]

function runScalableTranslationStatistics(rootDir::String; level::Integer=1, N::Integer=10000)
  # Get lib and model
  lib = joinpath(rootDir, "02_SourceModel", "02_Model", "01_AuthoringModel", "ScalableTranslationStatistics", "package.mo")
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"

  # Profile non-linear loops
  @showtime profilingInfo = profiling(modelName, lib; threshold=0)
  allProfiling = joinpath("profiling", "level$(string(level))")
  mkpath(allProfiling)
  @save joinpath(allProfiling, "profilingInfo.bson") profilingInfo

  # Find min-max values of used varaibles
  allUsedvars = vcat([prof.usingVars for prof in profilingInfo]...)
  @showtime (min, max) = minMaxValuesReSim(allUsedvars, modelName, lib)

  # Generate default FMU
  @showtime fmu = generateFMU(modelName, lib, workingDir = modelName)
  allFmuDir = joinpath("fmus", "level$(string(level))")
  mkpath(allFmuDir)
  cp(fmu, joinpath(allFmuDir, basename(fmu)), force=true)
  fmu = joinpath(allFmuDir, basename(fmu))

  # Generate extended FMU
  allEqs = [prof.eqInfo.id for prof in profilingInfo]
  @showtime fmu_interface = addEqInterface2FMU(modelName, fmu, allEqs, workingDir = modelName)
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

  #=
  allFmuDir = joinpath(@__DIR__, "fmus", "level$(string(level))")
  fmu_interface = joinpath(allFmuDir, modelName*".interface.fmu")
  allProfiling = joinpath(@__DIR__, "profiling", "level$(string(level))")
  @load joinpath(allProfiling, "profilingInfo.bson") profilingInfo
  profilingInfo = Array{ProfilingInfo}(profilingInfo)
  onnxFiles = String[]
  for prof in profilingInfo
    onnxModel = abspath(joinpath("onnx", "level$(string(level))", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)
  end
  =#

  # Include ONNX into FMU
  @showtime fmu_onnx = buildWithOnnx(fmu_interface,
                                     modelName,
                                     profilingInfo,
                                     onnxFiles;
                                     tempDir=modelName)
  basename(fmu_onnx)
  cp(fmu_onnx, joinpath(allFmuDir, basename(fmu_onnx)), force=true)
  fmu_onnx = joinpath(allFmuDir, basename(fmu_onnx))

  #return (profilingInfo, fmu, fmu_onnx)
end

function DataFrame(result::FMICore.FMU2Solution{FMU2}, names::Array{String})
  df = DataFrames.DataFrame(time=result.values.t)
  for i in 1:length(result.values.saveval[1])
    df[!, Symbol(names[i])] = [val[i] for val in result.values.saveval]
  end
  return df
end

function simulateFMU(pathToFMU, resultFile)
  mkpath(dirname(resultFile))

  outputVars = ["outputs[1]", "outputs[2]", "outputs[3]", "outputs[4]", "outputs[5]", "outputs[6]", "outputs[7]", "outputs[8]"]

  fmu = fmiLoad(pathToFMU)
  result = fmiSimulate(fmu, 0.0, 10.0; recordValues=outputVars)
  df = DataFrame(result, outputVars)

  bench = @benchmark fmiSimulate($fmu, 0.0, 10.0; recordValues=$outputVars)
  fmiUnload(fmu)

  CSV.write(resultFile, df)
  return bench
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
function genFMUs(levels=1:6)
  for level in levels
    @info "Starting level $level"
    redirect_stdio(stdout="level$(string(level)).log", stderr="level$(string(level)).log") do
      @time runScalableTranslationStatistics(rootDir, level = level)
    end
  end
end

# Test ONNX FMUs and bench times
function runBenchmarks(levels=1:6)
  for level in levels
    @info "Benchmarking level $level"
    resultDir = joinpath(@__DIR__, "results", "level$(string(level))")
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"
    defaultFMU = joinpath(@__DIR__, "fmus", "level$(string(level))", modelName*".fmu")
    onnxFMU = joinpath(@__DIR__, "fmus", "level$(string(level))", modelName*".onnx.fmu")
    defaultResult = joinpath(resultDir, modelName*".csv")
    onnxResult = joinpath(resultDir, modelName*".onnx.csv")
    redirect_stdio(stdout="bench_level$(string(level)).log", stderr="level$(string(level)).log") do
      time_default = simulateFMU(defaultFMU, defaultResult)
      time_onnx = simulateFMU(onnxFMU, onnxResult)
      @save joinpath(resultDir, "benchTimes.bson") time_default time_onnx
    end
  end
end

# Plot FMU simulations
function plotResult(;level=1, outputs=1:8, tspan=nothing)
  resultDir = joinpath(@__DIR__, "results", "level$(string(level))")
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"
  defaultResult = joinpath(resultDir, modelName*".csv")
  onnxResult = joinpath(resultDir, modelName*".onnx.csv")

  df_def = DataFrames.DataFrame(CSV.File(defaultResult))
  df_onnx = DataFrames.DataFrame(CSV.File(onnxResult))

  if tspan !== nothing
    i_start = findfirst(x->x>= tspan[1], df_def.time)
    i_end = findnext(x->x>= tspan[end], df_def.time, i_start)
    df_def = df_def[i_start:i_end,:]
    i_start = findfirst(x->x>= tspan[1], df_onnx.time)
    i_end = findnext(x->x>= tspan[end], df_onnx.time, i_start)
    df_onnx = df_onnx[i_start:i_end,:]
  end

  p = plot(title = "ScaledNLEquations.NLEquations_$(sizes[level])", legend=:topleft, xlabel="time [s]")
  for (i,out) in enumerate(outputs)
    name = "outputs[$out]"
    p = plot(p, df_def.time, df_def[!,Symbol(name)], label=name*" ref", color=i)
    p = plot(p, df_onnx.time, df_onnx[!,Symbol(name)], label=name*" onnx", color=i, linestyle=:dash)
  end

  return p
end


function animateData(;level=4, eq_idx=1, i_out=1, angle=40)
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"

  @load joinpath("profiling", "level$(string(level))", "profilingInfo.bson") profilingInfo
  prof = profilingInfo[eq_idx]

  dataFile = joinpath(@__DIR__, "data", "level$(string(level))", "eq_$(prof.eqInfo.id).csv")
  df = DataFrames.DataFrame(CSV.File(dataFile))

  inputNames = prof.usingVars
  outputNames = prof.iterationVariables

  p = scatter(df[!,Symbol(inputNames[1])],
              df[!,Symbol(inputNames[2])],
              df[!,Symbol(outputNames[i_out])],
              xlabel=inputNames[1], ylabel=inputNames[2], label=[outputNames[i_out]],
              title="Equation $(prof.eqInfo.id)",
              markersize=1, markeralpha=0.5, markerstrokewidth=0,
              color=2,
              camera = (angle, 30))

  return p
end

function genGif(level, i_out)
  anim = @animate for angle in 0:0.5:360
    animateData(;level=level, eq_idx=1, i_out=i_out, angle=angle)
  end
  gif(anim, "data_animation_level$(level)_var$(i_out).gif", fps = 30)
end
