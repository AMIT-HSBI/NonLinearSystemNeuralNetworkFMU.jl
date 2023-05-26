using DrWatson
@quickactivate "ScalableTranslationStatistics"

using CSV
using DataFrames
using CairoMakie
using ColorSchemes

sizes = [5,10,20,40]

csvFile = "$(@__DIR__)/../simTimes.csv"

function plotSimTimes(sizes, csvFile; filetype="svg")
  df = CSV.read(csvFile, DataFrame; ntasks=1)

  fig = Figure()
  ax = Axis(fig[1,1],
            title = "Simulation time (lower is better)",
            xticks = (sizes, ["N=$size" for size in sizes]),
            xscale = Makie.log2,
            #yscale = Makie.log10,
            ylabel = "time [s]")

  dflabels = ["ref", "onnxErr"]
  plotLabels = ["Newton-Raphson (NLS)", "Surrogate, error ctrl τ=1e-4 (NLS)"]

  #colors = ColorSchemes.viridis[range(0,1,length(dflabels))]
  colors = Makie.wong_colors()
  colNames = names(df)

  eqLabels = convert(Array{String}, df[!,"labels"])
  rowIdx = findfirst(x -> x==("sum"), eqLabels)

  for (i, label) in enumerate(dflabels)
    colIdx = findall(x -> endswith(x, label), colNames)
    lines!(ax, sizes, Array(df[rowIdx, colIdx])./1000, label=plotLabels[i], color=colors[i])
  end

  rowIdx = findfirst(x -> x==("total"), eqLabels)
  plotLabels = ["Newton-Raphson (total)", "Surrogate, error ctrl τ=1e-4 (total)"]

  for (i, label) in enumerate(dflabels)
    colIdx = findall(x -> endswith(x, label), colNames)
    lines!(ax, sizes, Array(df[rowIdx, colIdx])./1000, label=plotLabels[i], linestyle=:dash, color=colors[i])
  end

  axislegend(ax, position=:lt, margin=(20, 20, 20, 20))
  resize_to_layout!(fig)

  savename = plotsdir("Surrogat_vs_Reference.$(filetype)")
  save(savename, fig)
  return fig
end

plotSimTimes(sizes, csvFile; filetype="svg")
