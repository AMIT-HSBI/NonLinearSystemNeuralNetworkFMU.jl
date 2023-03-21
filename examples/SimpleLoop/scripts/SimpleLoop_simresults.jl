using DrWatson
@quickactivate "SimpleLoop"

using NonLinearSystemNeuralNetworkFMU
using Revise
using DataFrames
using CSV
using CairoMakie

# Model and plot parameters
b = -0.5
Ns = [100, 500, 750, 1000]

FH_Colors = ["#009BBB",
             "#E98300",
             "#C50084",
             "#722EA5",
             "#A2AD00"]

# Simulate fmus
sol = DataFrame[]
for N in Ns
  local workingDir = datadir("sims", "fmus")
  local fmu = datadir(workingDir, "$(modelName).onnx_N$(N).fmu")

  resultFile = "simpleLoop_onnx_res_N$(N).csv"
  logFile = joinpath(workingDir, modelName*"_OMSimulator_N$(N).log")
  cmd = `OMSimulator --resultFile=$resultFile --stopTime=2 "$(fmu)"`
  redirect_stdio(stdout=logFile, stderr=logFile) do
    run(Cmd(cmd, dir=workingDir))
  end

  push!(sol, CSV.read(joinpath(workingDir, resultFile), DataFrame; ntasks=1))
end

# Plot results
function plotSolution(dfs::Array{DataFrame}, Ns::Array{Int})

  fig = Figure(fontsize = 21,
               resolution = (1600, 800))

  axis = Axis(fig[1,1],
              title = "SimpleLoop Simulation Result",
              xlabel = "time t",
              ylabel = "y(t)")

  lines!(axis, dfs[1].time, dfs[1].y_ref, color=:black)

  for (i,df) in enumerate(dfs)
    lines!(axis, df.time, df.y, color=FH_Colors[i+1], linestyle = :dash)
  end

  l1 = LineElement(color = :black)
  l2 = LineElement(color = :black, linestyle = :dash)

  Legend(fig[1,1],
         [l1, l2],
         ["Ground truth", "Surroage"],
         tellheight = false,
         tellwidth = false,
         margin = (50, 50, 50, 50),
         halign = :left, valign = :top,
         labelsize = 21)

  groupN = [LineElement(color = FH_Colors[i+1], linestyle=:dash) for (i,_) in enumerate(dfs)]

  Legend(fig[1,2],
         groupN,
         string.(Ns),
         "N",
         labelsize = 21)

  return fig
end

fig = plotSolution(sol, Ns)
display(fig)
if !isdir(plotsdir())
  mkpath(plotsdir())
end
save(plotsdir("SimpleLoop_simresults_y.svg"), fig)
