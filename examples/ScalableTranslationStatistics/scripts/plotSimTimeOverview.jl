using DrWatson
@quickactivate "ScalableTranslationStatistics"

using CSV
using DataFrames
using CairoMakie
using ColorSchemes

function plotSimTimes(sizes, csvFile; plotSpeedup::Bool=true, filetype="svg")
  df = CSV.read(csvFile, DataFrames.DataFrame; ntasks=1)

  lenSizes = length(sizes)
  fig = Figure(fontsize = 24,
               resolution = (800, 800))
  time_axis = Axis(fig[1,1],
                   #title = "Simulation time",
                   xticks = (1:lenSizes, ["N=$size" for size in sizes]),
                   ylabel = "time [s] (lower is better)")

  local speedup_axis
  if plotSpeedup
    speedup_axis = Axis(fig[1,1],
                        xticks = (1:lenSizes, ["N=$size" for size in sizes]),
                        ylabel = "speedup (higher is better)",
                        yaxisposition = :right,
                        yticklabelalign = (:left, :center),
                        xticklabelsvisible = false,
                        yticklabelsvisible = true,
                        ygridvisible = false,
                        xlabelvisible = false)
  end

  dflabels = ["ref", "onnxErr"]
  plotLabels = ["Newton-Raphson (NLS accumulated)", "Surrogate, error ctrl τ=1e-4 (NLS accumulated)"]

  #colors = ColorSchemes.viridis[range(0,1,length(dflabels))]
  colors = Makie.wong_colors()
  colNames = names(df)

  # Plot sum of NLS evaluations
  eqLabels = convert(Array{String}, df[!,"labels"])
  rowIdx = findfirst(x -> x==("sum"), eqLabels)

  for (i, label) in enumerate(dflabels)
    colIdx = findall(x -> endswith(x, label), colNames)
    lines!(time_axis, 1:lenSizes, Array(df[rowIdx, colIdx])./1000, label=plotLabels[i], color=colors[i])
  end

  # Plot total time of simulation
  rowIdx = findfirst(x -> x==("total"), eqLabels)
  plotLabels = ["Newton-Raphson (total)", "Surrogate, error ctrl τ=1e-4 (total)"]

  for (i, label) in enumerate(dflabels)
    colIdx = findall(x -> endswith(x, label), colNames)
    lines!(time_axis, 1:lenSizes, Array(df[rowIdx, colIdx])./1000, label=plotLabels[i], linestyle=:dash, color=colors[i])
  end
  colIdx = findall(x -> endswith(x, "ref"), colNames)
  ylims!(time_axis, 0, maximum(df[rowIdx, colIdx]) ./ 1000 .* 1.05)

  # Plot speedup of total simulation
  if plotSpeedup
    colIdx = findall(x -> endswith(x, "ref"), colNames)
    ref_times = Array(df[rowIdx, colIdx])
    colIdx = findall(x -> endswith(x, "onnxErr"), colNames)
    onnx_times = Array(df[rowIdx, colIdx])
    speedup = ref_times./onnx_times
    b = barplot!(speedup_axis, 1:lenSizes, speedup, bar_labels=:y, width=0.2, color=(colors[3], 0.75))
    ylims!(speedup_axis, 0, maximum(speedup).*1.25)
    linkxaxes!(time_axis, speedup_axis)
    le = PolyElement(color=b.color, strokecolor=b.strokecolor)
    Legend(fig[2,1], [le], ["Speedup (total)"],
           tellheight = true, tellwidth = false,
           framevisible = false,
           halign = :center, valign = :top, orientation = :horizontal)
  end

  # Add legend
  #Legend(fig[2,1], time_axis,
  #       framevisible = false,
  #       tellheight = true, tellwidth = false,
  #       halign = :left, valign = :top)
  axislegend(time_axis, position=:lt, labelsize = 21, margin=(20, 20, 20, 20))
  resize_to_layout!(fig)

  # Save file
  savename = plotsdir("Surrogat_vs_Reference.$(filetype)")
  save(savename, fig; px_per_unit = 2)
  return fig
end
