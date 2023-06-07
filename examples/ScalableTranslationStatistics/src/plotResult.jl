using BSON: @load
using CairoMakie
using Colors
using CSV
using DataFrames
using Interpolations

include(srcdir("util.jl"))

# Plot FMU simulations
"""
    plotResult(modelName, resultDir, outputVars; tspan=nothing, fullNames=false, title="")

Plot results from modelname.csv and modelname.onnx.csv.
Optional argument tspan to only plot values inside interval.
"""
function plotResult(referenceResult::String,
                    onnxResult::String,
                    outputVars::Array{String},
                    varName::String;
                    residualResults::Union{String, Nothing}=nothing,
                    tspan::Union{Nothing, Tuple{Number,Number}}=nothing,
                    fullNames::Bool=false,
                    plotAbsErr::Bool=true,
                    eqId::Union{Nothing,Integer}=nothing,
                    orientation=:horizontal)

  df_ref = CSV.read(referenceResult, DataFrames.DataFrame; ntasks=1)
  df_onnx = CSV.read(onnxResult, DataFrames.DataFrame; ntasks=1)
  local df_res
  if residualResults !== nothing
    df_res = CSV.read(residualResults, DataFrames.DataFrame; ntasks=1)
  end

  # Restrict data frames to tspan
  if tspan !== nothing
    i_start = findfirst(x->x>= tspan[1], df_ref.time)
    i_end = findnext(x->x>= tspan[end], df_ref.time, i_start)
    df_ref = df_ref[i_start:i_end,:]
    i_start = findfirst(x->x>= tspan[1], df_onnx.time)
    i_end = findnext(x->x>= tspan[end], df_onnx.time, i_start)
    df_onnx = df_onnx[i_start:i_end,:]
    if residualResults !== nothing
      i_start = findfirst(x->x>= tspan[1], df_res.time)
      i_end = findnext(x->x>= tspan[end], df_res.time, i_start)
      df_res = df_res[i_start:i_end,:]
    end
  end

  errorFunctions = Any[]
  for var in outputVars
    func_def = linear_interpolation(df_ref.time, df_ref[!,Symbol(var)])
    func_onnx = linear_interpolation(df_onnx.time, df_onnx[!,Symbol(var)])
    f(t) = abs(func_def(t) - func_onnx(t))  # Absolute error
    #f(t) = abs(func_def(t) - func_onnx(t))/(max(abs(func_def(t)), abs(func_onnx(t)), 1e-2)) # Relative difference with max
    #f(t) = abs(func_def(t) - func_onnx(t))/(abs(func_def(t))) # Relative error
    push!(errorFunctions, f)
  end
  # Compute error
  function relativeError(t, i)
    return errorFunctions[i](t)
  end

  colors = distinguishable_colors(length(outputVars)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

  fig1 = Figure(fontsize = 24,
                resolution = (800, 800))
  grid_simResults = fig1[1, 1:2] = GridLayout()
  grid_simError = fig1[2, 1:2] = GridLayout()
  grid_legend = fig1[3, 1:2] = GridLayout()

  ax_simResults = Axis(grid_simResults[1,1],
                       title = "$varName variables",
                       xlabel = "time [s]",
                       xticks = 0:1:10,
                       xminorticks = IntervalsBetween(2),
                       xminorticksvisible = true,
                       xminorgridvisible = true,
                       ylabel = "Relative position [m]",
                       yminorticksvisible = true,
                       height = 300)
  ax_simError = Axis(grid_simError[1,1],
                     title = "Difference surrogate and reference",
                     xlabel = "time [s]",
                     xticks = 0:1:10,
                     xminorticks = IntervalsBetween(2),
                     xminorticksvisible = true,
                     xminorgridvisible = true,
                     yminorticksvisible = true,
                     height = 300)

  for (i, var) in enumerate(outputVars)
    lines!(ax_simResults, df_ref.time, df_ref[!,Symbol(var)], color=colors[i])
    lines!(ax_simResults, df_onnx.time, df_onnx[!,Symbol(var)], color=colors[i], linestyle=:dash)
    # Error plots
    if fullNames
      label = join(split(var, ".")[2:end], ".")
    else
      label = last(split(var, "."))
    end
    lines!(ax_simError, df_ref.time, relativeError.(df_ref.time, i), label=label, color=colors[i])
  end

  # Legend
  elem_1 = LineElement(color = :black, linestyle = nothing)
  elem_2 = LineElement(color = :black, linestyle = :dash)
  axislegend(ax_simResults,
             [elem_1, elem_2],
             ["reference", "surrogate"],
             position = :lt)

  Legend(grid_legend[1,1], ax_simError,
         tellwidth=true, tellheight=true,
         orientation = orientation)
  resize_to_layout!(fig1)

  # Optional residual plot
  local fig2
  if residualResults !== nothing
    fig2 = Figure(fontsize = 24,
                  resolution = (800, 400))
    grid_residual = fig2[1, 1:2] = GridLayout()

    ax_residual = Axis(grid_residual[1,1],
                       title = "Residual",
                       xlabel = "time [s]",
                       xticks = 0:1:10,
                       xminorticks = IntervalsBetween(10),
                       xminorticksvisible = true,
                       xminorgridvisible = true,
                       yminorticksvisible = true,
                       height = 300)

    for (i, _) in enumerate(outputVars)
      lines!(ax_residual, df_res.time, df_res[!,Symbol("res[$(i-1)]")], color=colors[i], label=L"$f_{res_%$(i)}$")
    end
    #lines!(ax_residual, df_res.time, df_res[!,:rel_error], color=colors[length(outputVars)+1], label=L"\tau_{rel}(f_{res})")
    lines!(ax_residual, df_res.time, df_res[!,:scaled_res_norm], color=colors[length(outputVars)+1], label=L"||\tau_s(J)||_2")
    lines!(ax_residual, df_res.time[[1,length(df_res.time)]], [1.0, 1.0] , color=:grey, linestyle=:dash)

    axislegend(ax_residual, orientation = :horizontal, position = :lt)
    resize_to_layout!(fig2)
  end

  if residualResults !== nothing
    return fig1, fig2
  else
    return fig1
  end
end

function plotTrainDataHistogram(vars::Array{String}, df_trainData::DataFrames.DataFrame; title = "")
  aspectRatio = 1.0
  nCols = Integer(ceil(sqrt(length(vars)*aspectRatio)))
  nRows = Integer(ceil(length(vars)/nCols))

  fig = Figure(fontsize = 32,
               resolution = (nRows*800, nCols*600))

  Label(fig[0,:], text=title, fontsize=32, tellwidth=false, tellheight=true)
  grid = GridLayout(nRows, nCols; parent = fig)

  row = 1
  col = 1
  for (i,var) in enumerate(vars)
    axis = Axis(grid[row, col],
                xlabel = join(split(var, ".")[2:end], '.'),
                ylabel = "# samples")

    CairoMakie.hist!(axis,
                     df_trainData[!, var],
                     bins = 15,
                     normalization = :none)

    if i%nCols == 0
      row += 1
      col = 1
    else
      col += 1
    end
  end

  for row in 1:nRows
    idx = (row-1)*nCols + 2
    for i in 1:nCols-1
      linkyaxes!(fig.content[idx], fig.content[idx+i])
      hideydecorations!(fig.content[idx+i], grid = false)
    end
  end
  fig[1,1] = grid

  return fig
end

function plotLoss(lossFile)
  df = CSV.read(lossFile, DataFrames.DataFrame; ntasks=1)

  fig = Figure(fontsize = 42)
  axis = Axis(fig[1, 1],
              xlabel = "Epochs",
              yscale = Makie.log10,
              ylabel = "Loss",
              yminorticksvisible = true, yminorgridvisible = true,
              yminorticks = IntervalsBetween(10))

  lines!(axis,
         df.epoch,
         df.lossTrain,
         linewidth = 3,
         label = "training")

  lines!(axis,
         df.epoch,
         df.lossTest,
         linewidth = 3,
         label = "test")

  axislegend(axis, position = :rt, margin = (20, 20, 20, 20))
  colsize!(fig.layout, 1, Aspect(1, 3.0))
  resize_to_layout!(fig)
  return fig
end
