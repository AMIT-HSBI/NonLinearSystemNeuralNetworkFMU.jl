using DrWatson
@quickactivate "SimpleLoop_ActiveLearning"

import NonLinearSystemNeuralNetworkFMU
import LinearAlgebra

include(srcdir("main.jl"))
include(srcdir("activeLearn.jl"))
include(srcdir("simulateFMU.jl"))

modelName = "simpleLoop"

df = DataFrames.DataFrame(y_err=Float64[], res_err=Float64[])

N = 1400
p = 1.0
for i in 0:9
  @info "N=$N p=$p i=$i"
  mymain(modelName, N; pretrain=p)
  fname = joinpath(datadir("sims", "$(modelName)_$(N)_$(p)"), "temp-omsimulator/$(modelName)_onnx_res")
  mv(fname*".csv",fname*"_$i.csv")
  result = CSV.read(fname*"_$i.csv", DataFrames.DataFrame; ntasks=1)
  y_err = LinearAlgebra.norm(result[:, :y_err])
  res_err = LinearAlgebra.norm(result[:, Symbol("res_err[1]")])
  push!(df, vcat(y_err, res_err))
end

CSV.write("comparison_$(N)_$(p).csv", df)
