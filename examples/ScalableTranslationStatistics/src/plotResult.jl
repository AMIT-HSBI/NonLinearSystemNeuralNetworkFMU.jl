using BSON: @load
using CairoMakie
using Colors
using CSV
using DataFrames
using Interpolations


# Plot FMU simulations
"""
    plotResult(modelName, resultDir, outputVars; tspan=nothing)

Plot results from modelname.csv and modelname.onnx.csv.
Optional argument tspan to only plot values inside interval.
"""
function plotResult(defaultResult, onnxResult, outputVars::Array{String}; tspan::Union{Nothing, Tuple{Number,Number}}=nothing)

  df_def = DataFrames.DataFrame(CSV.File(defaultResult))
  df_onnx = DataFrames.DataFrame(CSV.File(onnxResult))

  if tspan !== nothing
    i_start = findfirst(x->x>= tspan[1], df_def.time)
    i_end = findnext(x->x>= tspan[end], df_def.time, i_start)
    df_def = df_def[i_start:i_end,:]
    i_start = findfirst(x->x>= tspan[1], df_onnx.time)
    i_end = findnext(x->x>= tspan[end], df_onnx.time, i_start)
    df_onnx = df_onnx[i_start:i_end,:]
  end


  errorFunctions = Any[]
  for var in outputVars
    func_def = linear_interpolation(df_def.time, df_def[!,Symbol(var)])
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


  colors = distinguishable_colors(length(outputVars), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

  fig = Figure(fontsize = 18,
               resolution = (800, 800))
  grid_top = fig[1, 1:2] = GridLayout()
  grid_bottom = fig[2, 1:2] = GridLayout()
  grid_right = fig[1:2, 3] = GridLayout()
  ax_top = Axis(grid_top[1,1],
            title = "Simulation results",
            xlabel = "time [s]",
            ylabel = "Relative position [m]")
  ax_bottom = Axis(grid_bottom[1,1],
                   title = "Error",
                   xlabel = "time [s]",
                   ylabel = L"Absolute error $\epsilon_{abs}$")
  for (i, var) in enumerate(outputVars)
    lines!(ax_top, df_def.time, df_def[!,Symbol(var)], color=colors[i])
    lines!(ax_top, df_onnx.time, df_onnx[!,Symbol(var)], color=colors[i], linestyle=:dash)
    # Error plots
    lines!(ax_bottom, df_def.time, relativeError.(df_def.time, i), label=var, color=colors[i])
  end
  elem_1 = LineElement(color = :black, linestyle = nothing)
  elem_2 = LineElement(color = :black, linestyle = :dash)
  axislegend(ax_top,
             [elem_1, elem_2],
             ["reference", "surrogate"],
             position = :lt)
  Legend(grid_right[1,1], ax_bottom)
  return fig
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
