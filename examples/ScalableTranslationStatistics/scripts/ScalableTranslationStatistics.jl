# $ julia -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()"
# $ nohup julia -e "include(\"ScalableTranslationStatistics.jl\"); genFMUs()" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

#using Revise

#using BSON: @save
#using FMI
#using FMICore
#using CSV
#using DataFrames
#using BenchmarkTools

include(srcdir("genSurrogates.jl"))

# Specify location of ScalableTranslationStatistics library
rootDir = "/mnt/home/aheuermann/workdir/phymos/ScalableTranslationStatistics"
modelicaLib = joinpath(rootDir, "02_SourceModel", "02_Model", "01_AuthoringModel", "ScalableTranslationStatistics", "package.mo")

@assert haskey(ENV, "ORT_DIR") "Environment variable ORT_DIR not set!"

sizes = [5, 10, 20, 40, 80, 160]

# Test ONNX FMUs and bench times
function runBenchmarks(sizes::Array{Int}, modelicaLib::String; N::Int=1000)

  @assert isfile(modelicaLib) "Couldn't find Modelica file '$(modelicaLib)'"

  for size in sizes
    @info "Benchmarking size $size"
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"

    @info "Generating fmu and onnx.fmu"
    local profilingInfo
    local fmu
    local fmu_onnx
    logFile = datadir("sims", split(modelName, ".")[end] * ".log")
    redirect_stdio(stdout=logFile, stderr=logFile) do
      @time runScalableTranslationStatistics(modelicaLib, modelName, size=size, N=N)
    end

  end
end

runBenchmarks(sizes, modelicaLib; N=100)
