using CairoMakie
using DataFrames


function plotResults(vars::Array{String}, df_ref::DataFrame, df_onnx::Union{DataFrame,Nothing}; title="", epsilon = 0.01)
  nRows = Integer(ceil(sqrt(length(vars))))

  fig = Figure(fontsize = 32,
               resolution = (nRows*800, nRows*600),
               title = title)

  row = 1
  col = 1
  for (i,var) in enumerate(vars)
    axis = Axis(fig[row, col],
                xlabel = "time",
                ylabel = var)
    # SimResult: Reference solution
    CairoMakie.lines!(axis,
                      df_ref.time, df_ref[!, var],
                      label = "ref")
    # ϵ tube
    CairoMakie.lines!(axis,
                      df_ref.time, epsilonTube(df_ref[!, var], epsilon),
                      color = :seagreen,
                      linestyle = :dash)

    CairoMakie.lines!(axis,
                      df_ref.time, epsilonTube(df_ref[!, var], -epsilon),
                      color = :seagreen,
                      linestyle = :dash,
                      label = "ϵ: ±$epsilon")

    # SimResult: ONNX Surrogate
    if df_onnx !== nothing
      CairoMakie.lines!(axis,
                        df_onnx.time, df_onnx[!, var],
                        color = :orangered1,
                        linestyle = :dashdot,
                        label = "surrogate")
    end

    axislegend()

    if i%nRows == 0
      row += 1
      col = 1
    else
      col += 1
    end
  end

  return fig
end


function epsilonTube(values::Vector{Float64}, ϵ=0.1)
  nominal = 0.5* (minimum(abs.(values)) + maximum(abs.(values)))
  return values .+ ϵ*nominal
end
