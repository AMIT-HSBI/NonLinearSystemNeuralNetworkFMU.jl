using Revise
using CSV
using DataFrames
using BSON: @load
using CairoMakie
using Colors
using Interpolations


# Plot FMU simulations
"""
    plotResult(modelName, resultDir, outputVars; tspan=nothing)

Plot results from modelname.csv and modelname.onnx.csv.
Optional argument tspan to only plot values inside interval.
"""
function plotResult(modelName, resultDir, outputVars::Array{String}; tspan::Union{Nothing, Tuple{Number,Number}}=nothing)
  defaultResult = joinpath(resultDir, modelName*".csv")
  onnxResult = joinpath(resultDir, modelName*".onnx.csv")

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
  save("$(modelName)_results.pdf", fig, px_per_unit=2)
  #return fig
end

#=

function animateData(;level=4, eq_idx=1, i_out=1, angle=40)
  @load joinpath("level$(string(level))", "profilingInfo.bson") profilingInfo
  prof = profilingInfo[eq_idx]

  dataFile = joinpath(@__DIR__, "level$(string(level))", "data", "eq_$(prof.eqInfo.id).csv")
  df = DataFrames.DataFrame(CSV.File(dataFile))

  inputNames = prof.usingVars
  outputNames = prof.iterationVariables

  p = scatter(df[!,Symbol(inputNames[1])],
              df[!,Symbol(inputNames[2])],
              df[!,Symbol(outputNames[i_out])],
              xlabel=inputNames[1], ylabel=inputNames[2], label=[outputNames[i_out]],
              title="Equation $(prof.eqInfo.id)",
              markersize=1, markeralpha=0.5, markerstrokewidth=0,
              color=2,
              camera = (angle, 30))

  return p
end


function genGif(level, i_out)
  anim = @animate for angle in 0:0.5:360
    animateData(;level=level, eq_idx=1, i_out=i_out, angle=angle)
  end
  gif(anim, "data_animation_level$(level)_var$(i_out).gif", fps = 30)
end
=#