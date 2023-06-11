using DrWatson
@quickactivate "SimpleLoop_ActiveLearning"

import NonLinearSystemNeuralNetworkFMU
import LinearAlgebra

include(srcdir("main.jl"))
include(srcdir("activeLearn.jl"))
include(srcdir("simulateFMU.jl"))

modelName = "simpleLoop"

df = DataFrames.DataFrame(N=Int64[], p=Float64[], y_max=Float64[], y_mse=Float64[], res_max=Float64[], res_mse=Float64[])

repeats = 10
for N in 600:200:1400
  for p in 0.0:0.25:1.0
    y_max = y_mse = res_max = res_mse = 0.0
    anzahl = 0
    for i in 0:repeats-1
      fname = joinpath(datadir("sims", "$(modelName)_$(N)_$(p)"), "temp-omsimulator/$(modelName)_onnx_res_$i.csv")
      result = CSV.read(fname, DataFrames.DataFrame; ntasks=1)
      y_max = max(y_max, maximum(abs.(result[:, :y_err])))
      y_mse += sum(result[:, :y_err].^2)/length(result[:, :y_err])
      #res_max = max(res_max, maximum(abs.(result[:, Symbol("res_err[1]")])))
      res_max += maximum(abs.(result[:, Symbol("res_err[1]")]))
      res_mse += sum(result[:, Symbol("res_err[1]")].^2)/length(result[:, Symbol("res_err[1]")])
      anzahl += 1
    end
    push!(df, vcat(N, p, y_max, sqrt(y_mse/anzahl), res_max/anzahl, sqrt(res_mse/anzahl)))
  end
end

CSV.write("comparison.csv", df)
