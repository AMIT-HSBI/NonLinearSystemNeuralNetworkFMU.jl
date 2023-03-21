using DrWatson
@quickactivate "SimpleLoop"

using NonLinearSystemNeuralNetworkFMU
using DataFrames
using CSV

include(srcdir("plotData.jl"))
include(srcdir("isRight.jl"))

# Model and plot parameters
N = 100
b = -0.5

FH_Colors = ["#009BBB",
             "#E98300",
             "#C50084",
             "#722EA5",
             "#A2AD00"]

# Genrate data for SimpleLoop
modelName = "simpleLoop"
moFiles = [abspath(srcdir(),"simpleLoop.mo")]
workingDir = datadir("sims", modelName*"_$N")
options = NonLinearSystemNeuralNetworkFMU.OMOptions(workingDir=workingDir)

# Generate data
(csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; options=options, reuseArtifacts=false, N=100)

# Plot data and save svg
df =  CSV.read(csvFiles[1], DataFrame; ntasks=1)
fig = plotData(df)
display(fig)
mkpath(dirname(plotsdir("SimpleLoop_data.svg")))
save(plotsdir("SimpleLoop_data.svg"), fig)

df_filtered = filter(row -> !isRight(row.s, row.r, row.y; b=b), df)
fig = plotData(df_filtered; title = "SimpleLoop: Training Data (filtered)")
display(fig)
mkpath(dirname(plotsdir("SimpleLoop_data_filtered.svg")))
save(plotsdir("SimpleLoop_data_filtered.svg"), fig)
