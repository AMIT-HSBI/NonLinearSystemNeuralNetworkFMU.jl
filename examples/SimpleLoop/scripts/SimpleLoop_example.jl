using DrWatson
@quickactivate "SimpleLoop"

using NonLinearSystemNeuralNetworkFMU
using NaiveONNX
using Revise
using DataFrames
using CSV
using CairoMakie

# Changable parameters
N = 100

# Model and plot parameters
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

# Clean up
#rm(joinpath(workingDir), recursive=true, force=true)

# Generate data
(csvFiles, fmu, profilingInfo) = NonLinearSystemNeuralNetworkFMU.main(modelName, moFiles; options=options, reuseArtifacts=false, N=N)


"""
Filter data for `x = r*s + b -y <= y` to get uniqe data points.
"""
function isRight(s,r,y; b)
  x = r*s + b -y
  return x > y
end

# Plot data
function plotData(df::DataFrame; title = "SimpleLoop: Training Data")
  fig = Figure(fontsize = 18,
               resolution = (800, 800))

  axis = Axis(fig[1,1],
              title = title,
              xlabel = "x",
              ylabel = "y")

  limits!(axis, -2.5, 3.5, -2.5, 3.5)

  # Expected solutions
  X1 = 0.5 .* ( -sqrt.(-b^2 .- 2*b .* df.r .* df.s - (df.r).^2 .* ((df.s).^2 .- 2)) .+ b .+ df.r .* df.s)
  Y1 = df.r .* df.s .+ b .- X1
  s1 = scatter!(axis, X1, Y1, color=(FH_Colors[1], 0.5), markersize = 21)

  X1 = 0.5 .* ( sqrt.(-b^2 .- 2*b .* df.r .* df.s - (df.r).^2 .* ((df.s).^2 .- 2)) .+ b .+ df.r .* df.s)
  Y1 = df.r .* df.s .+ b .- X1
  s2 = scatter!(axis, X1, Y1, color=(FH_Colors[2], 0.5), markersize = 21)

  # Generated data
  X_train = df.r.*df.s .+ b .- df.y;
  s3 = scatter!(axis, X_train, df.y, color=FH_Colors[3], markersize=12, marker=:xcross)

  marker = [MarkerElement(color=(FH_Colors[1], 0.5), marker = :circle, markersize = 21, points = Point2f[(0, 0.5)]),
            MarkerElement(color=(FH_Colors[2], 0.5), marker = :circle, markersize = 21, points = Point2f[(0.75, 0.5)])]

  Legend(fig[1,1],
         [marker, s3],
         ["Expected", "Generated"],
         tellheight = false,
         tellwidth = false,
         margin = (50, 50, 50, 50),
         halign = :left, valign = :bottom,
         labelsize = 21)

  return fig
end

# Save plots
df =  CSV.read(csvFiles[1], DataFrame)
fig = plotData(df)
save(plotsdir("SimpleLoop_data.svg"), fig)

df_filtered = filter(row -> !isRight(row.s, row.r, row.y; b=b), df)
fig = plotData(df_filtered; title = "SimpleLoop: Training Data (filtered)")
save(plotsdir("SimpleLoop_data_filtered.svg"), fig)
