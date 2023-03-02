# $ julia --threads=auto -e "include(\"ScalableTranslationStatistics.jl\")"
# $ nohup julia --threads=auto -e "include(\"ScalableTranslationStatistics.jl\")" &

using DrWatson
@quickactivate "ScalableTranslationStatistics"

include(srcdir("genSurrogates.jl"))

# Specify location of ScalableTranslationStatistics library
rootDir = "/mnt/home/aheuermann/workdir/phymos/ScalableTranslationStatistics"
modelicaLib = joinpath(rootDir, "02_SourceModel", "02_Model", "01_AuthoringModel", "ScalableTranslationStatistics", "package.mo")

@assert haskey(ENV, "ORT_DIR") "Environment variable ORT_DIR not set!"


"""
Generate surroagate FMUs for Modelica model.
"""
function genAllSurrogates(sizes::Array{Int}, modelicaLib::String; N::Int=1000)

  @assert isfile(modelicaLib) "Couldn't find Modelica file '$(modelicaLib)'"

  for size in sizes
    @info "Benchmarking size $size"
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"

    @info "Generating fmu and onnx.fmu"
    logFile = datadir("sims", split(modelName, ".")[end] * ".log")
    mkpath(dirname(logFile))
    redirect_stdio(stdout=logFile, stderr=logFile) do
      @time genSurrogate(modelicaLib, modelName; N=N)
    end
  end
end

sizes = [5, 10, 20, 40, 80]
genAllSurrogates(sizes, modelicaLib; N=100)
