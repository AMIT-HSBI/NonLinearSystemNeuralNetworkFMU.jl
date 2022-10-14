using Plots
using CSV
using DataFrames
using BSON: @load
using GLMakie

# Plot FMU simulations
function plotResult(;level=1, outputs=1:8, tspan=nothing)
  resultDir = joinpath(@__DIR__, "results", "level$(string(level))")
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[level])"
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

  p = plot(title = "ScaledNLEquations.NLEquations_$(sizes[level])", legend=:topleft, xlabel="time [s]")
  for (i,out) in enumerate(outputs)
    name = "outputs[$out]"
    p = plot(p, df_def.time, df_def[!,Symbol(name)], label=name*" ref", color=i)
    p = plot(p, df_onnx.time, df_onnx[!,Symbol(name)], label=name*" onnx", color=i, linestyle=:dash)
  end

  return p
end


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
