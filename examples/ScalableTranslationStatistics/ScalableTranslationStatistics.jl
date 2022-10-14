# $ julia -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()"
# $ nohup julia -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()" &

cd(@__DIR__)
using NaiveONNX
import Pkg; Pkg.activate("../..")
using NonLinearSystemNeuralNetworkFMU
using BSON: @save
using FMI
using FMICore
using CSV
using DataFrames
using BenchmarkTools

ENV["ORT_DIR"] = "/mnt/home/aheuermann/workdir/julia/NonLinearSystemNeuralNetworkFMU.jl/onnxruntime-linux-x64-1.12.1"
rootDir = "/mnt/home/aheuermann/workdir/phymos/ScalableTranslationStatistics"

sizes = ["5", "10", "20", "40", "80", "160"]

function runScalableTranslationStatistics(rootDir::String; level::Integer=1, N::Integer=10000)
  # Get lib and model
  lib = joinpath(rootDir, "02_SourceModel", "02_Model", "01_AuthoringModel", "ScalableTranslationStatistics", "package.mo")
  if !isfile(lib)
    @error "Could not find Modelica library at $(lib)"
  end
  moFiles = [lib]
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"
  workdir = joinpath(@__DIR__, "level$level")

  # Generate training data
  (csvFiles, fmu, profilingInfo) = main(modelName, moFiles, workdir=workdir, reuseArtifacts=true, N=N)

  # Train ONNX
  onnxFiles = String[]
  for (i ,prof) in enumerate(profilingInfo)
    onnxModel = abspath(joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx"))
    push!(onnxFiles, onnxModel)
    nInputs = length(prof.usingVars)

    @showtime trainONNX(csvFiles[i], onnxModel, nInputs; nepochs=100, losstol=1e-4)
  end

  # Include ONNX into FMU
  fmu_interface = joinpath(@__DIR__, "level$level", modelName*".interface.fmu")
  @showtime fmu_onnx = buildWithOnnx(fmu_interface,
                                     modelName,
                                     profilingInfo,
                                     onnxFiles;
                                     tempDir=workdir)

  return (profilingInfo, fmu, fmu_onnx)
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

  bench = @benchmark fmiSimulate($fmu, 0.0, 10.0; recordValues=$outputVars) samples=5 seconds=120*5 evals=1
  fmiUnload(fmu)

  CSV.write(resultFile, df)
  return bench
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
    defaultFMU = joinpath(@__DIR__, "level$(string(level))", modelName*".fmu")
    onnxFMU = joinpath(@__DIR__, "level$(string(level))", modelName*".onnx.fmu")
    defaultResult = joinpath(resultDir, modelName*".csv")
    onnxResult = joinpath(resultDir, modelName*".onnx.csv")
    redirect_stdio(stdout="bench_level$(string(level)).log", stderr="level$(string(level)).log") do
      time_default = simulateFMU(defaultFMU, defaultResult)
      @show time_default
      time_onnx = simulateFMU(onnxFMU, onnxResult)
      @show time_onnx
      @save joinpath(resultDir, "benchTimes.bson") time_default time_onnx
    end
  end
end
