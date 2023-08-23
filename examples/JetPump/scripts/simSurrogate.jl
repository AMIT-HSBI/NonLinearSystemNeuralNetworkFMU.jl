using DrWatson
@quickactivate "JetPump"

using NonLinearSystemNeuralNetworkFMU
using Plots
using CSV
using DataFrames
include(srcdir("replaceSurrogate.jl"))

modelName = "Scenario_01"
n = 100_000
workdir = datadir("sims", "$(modelName)_$(n)")

# Create flat surrogate model
prof = getProfilingInfo(joinpath(workdir, "profilingInfo.bson"))[1]
eqIndex = prof.eqInfo.id
infoJsonFile = abspath(joinpath(workdir, "temp-profiling", modelName*"_info.json"))
flatModel = abspath(joinpath(workdir, "$(modelName)_flat.mo"))

surrogatEqnsMo = joinpath(workdir, "surrogatEqns.mo")
surrogatEqns = readlines(surrogatEqnsMo)
flatModelSurrogate = replaceSurrogates(flatModel, prof, infoJsonFile, eqIndex, surrogatEqns)

# Simulate models
for file in [flatModel, flatModelSurrogate]
  tempDir = joinpath(workdir, "temp-sim", splitext(basename(file))[1])
  mkpath(tempDir)
  omc = OMJulia.OMCSession()
  msg = sendExpression(omc, "loadFile(\"$(file)\")")
  if (msg != true)
    msg = sendExpression(omc, "getErrorString()")
    error("Failed to load file $(file)!", abspath(logFilePath))
  end
  sendExpression(omc, "cd(\"$(tempDir)\")")
  sendExpression(omc, "simulate($(modelName), outputFormat=\"csv\", simflags=\"-override=stopTime=100.0,intervals=1000\")")
  OMJulia.sendExpression(omc, "quit()",parsed=false)
end

# Plot results
refResultFile = joinpath(workdir, "temp-sim", splitext(basename(flatModel))[1], modelName*"_res.csv")
df_ref = DataFrame(CSV.File(refResultFile))
rename!(df_ref, replace.( names(df_ref), "'" => ""))

surrogateResultFile = joinpath(workdir, "temp-sim", splitext(basename(flatModelSurrogate))[1], modelName*"_res.csv")
df_sur = DataFrame(CSV.File(surrogateResultFile))
rename!(df_sur, replace.( names(df_ref), "'" => ""))

plot(df_ref[!,"time"], df_ref[!,"suctionFlow.T"], label="refference")
plot!(df_sur[!,"time"], df_sur[!,"suctionFlow.T"], label="surrogate")
xlabel!("time (s)")
ylabel!("T (K)")

plot(df_ref[!,"time"], df_ref[!,"suctionFlow.p"]./1000, label="refference")
plot!(df_sur[!,"time"], df_sur[!,"suctionFlow.p"]./1000, label="surrogate")
xlabel!("time (s)")
ylabel!("p (kPa)")

