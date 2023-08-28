using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using FMI
using Flux

modelName = "IEEE_14_Buses"
moFiles = [srcdir("IEEE_14_Buses.mo")]

function genData(modelName::String, moFiles::Array{String}, n::Integer)
  workdir = datadir("sims", "$(modelName)_$(n)")
  options = OMOptions(workingDir=workdir,
                      clean=false,
                      commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls")
  dataGenOptions = DataGenOptions(method=RandomMethod(), n=n, nBatches=Threads.nthreads())
  
  main(modelName, moFiles; omOptions=options, dataGenOptions=dataGenOptions, reuseArtifacts=true)
end

(csvFiles, fmu, profilingInfo) = genData(modelName, moFiles, 100)


fmu_from_string = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_100/IEEE_14_Buses.fmu")
fmu_comp = FMI.fmiInstantiate!(fmu_from_string; loggingOn=true)
#FMI.fmiInfo(fmu_from_string)

status = NonLinearSystemNeuralNetworkFMU.fmiEvaluateEq(fmu_comp, 1403)

(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, 1403, rand(Float64, 110))

