# $ julia --threads=auto -e "include(\"ScalableTranslationStatistics.jl\")"
# $ nohup julia --threads=auto -e "include(\"ScalableTranslationStatistics.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

include(srcdir("genSurrogates.jl"))

# Specify location of ScalableTranslationStatistics library
modelicaLib = joinpath(@__DIR__, "..", "ScalableTranslationStatistics", "package.mo")

@assert haskey(ENV, "ORT_DIR") "Environment variable ORT_DIR not set!"


"""
Generate surroagate FMUs for Modelica model.
"""
function genAllSurrogates(sizes::Array{Int}, modelicaLib::String; n::Int=1000, genData::Bool=true)

  @assert isfile(modelicaLib) "Couldn't find Modelica file '$(modelicaLib)'"

  for size in sizes
    @info "Benchmarking size $size"
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"

    @info "Generating fmu and onnx.fmu"
    logFile = datadir("sims", split(modelName, ".")[end] * ".log")
    mkpath(dirname(logFile))
    #redirect_stdio(stdout=logFile, stderr=logFile) do
      @time genSurrogate(modelicaLib, modelName; n=n, genData=genData)
    #end
  end
end
