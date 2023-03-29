using DrWatson
@quickactivate "IEEE14_proximity"

using BSON
using CairoMakie
using CSV
using DataFrames
using Flux
using NonLinearSystemNeuralNetworkFMU

"""
Plot trained area, a zoomed in version of trained area and the surrogate solution when integrated into the ODE.
"""
function plotStuff(modelName, N; fileType = "svg")
  workdir = datadir("sims", "$(modelName)_$(N)")
  plotdir = plotsdir("$(modelName)_$(N)")
  mkpath(plotdir)

  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  prof = Array{ProfilingInfo}(dict[first(keys(dict))])[1]

  ref_csv = joinpath(workdir, "temp-omsimulator", "$(modelName)_ref.csv")
  onnx_csv = joinpath(workdir, "temp-omsimulator", "$(modelName)_onnx_res.csv")
  trainData_csv = joinpath(workdir, "data", "eq_1403.csv")

  ref_results = CSV.read(ref_csv, DataFrame; ntasks=1)
  onnx_results = CSV.read(onnx_csv, DataFrame; ntasks=1)
  trainData = CSV.read(trainData_csv, DataFrame; ntasks=1)

  figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, df_trainData=trainData, title="Training Area", epsilon=0.1)
  savename = joinpath(plotdir, "trainArea.$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, df_trainData=trainData, title="Training Area t∈[0,0.1]", epsilon=0.1, tspan=(0.0, 0.1))
  savename = joinpath(plotdir, "trainAreaStart.$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.iterationVariables[1:4], ref_results; df_surrogate=onnx_results, title="Surrogate Results", epsilon=0.1)
  savename = joinpath(plotdir, "surrogateSolution.$(fileType)")
  save(savename, figure)
end

"""
Plot the solution of the ANN surrogate when using the reference solution as input,
so errors wont influence the next time step.
"""
function SurrogateRefRes(modelName, N; fileType="svg", proximity=1)
  @assert Threads.nthreads() == 1 "This breaks with multiple threads!"

  workdir = datadir("sims", "$(modelName)_$(N)")
  plotdir = plotsdir("$(modelName)_$(N)")
  mkpath(plotdir)

  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  prof = Array{ProfilingInfo}(dict[first(keys(dict))])[1]

  ref_csv = joinpath(workdir, "temp-omsimulator", "$(modelName)_ref.csv")
  ref_results = CSV.read(ref_csv, DataFrame; ntasks=1)

  BSON.@load joinpath(workdir, "onnx", "eq_1403.onnx.bson") model

  onnx_results = deepcopy(ref_results[!, vcat(prof.usingVars, prof.iterationVariables)])
  onnx_results[proximity+1:end, prof.iterationVariables] .= NaN

  for (i, _) in enumerate(eachrow(onnx_results))
    if i <= proximity
      continue
    end
    usingVars =  onnx_results[i, prof.usingVars]
    prev_solution = onnx_results[i-proximity, prof.iterationVariables]

    input = vcat(Array{Float32}(usingVars), Array{Float32}(prev_solution))

    onnx_results[i, prof.iterationVariables] = model(input)
  end

  t0 = ref_results[proximity, "time"]

  figure = plotTrainArea(prof.iterationVariables, ref_results; df_surrogate=onnx_results, title="ANN with inputs from ref sol, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  savename = joinpath(plotdir, "surrWithRefInputs_out_prox_$(proximity).$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.iterationVariables[2:5], ref_results; df_surrogate=onnx_results, title="ANN with inputs from reference solution, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  savename = joinpath(plotdir, "surrWithRefInputs_4_prox_$(proximity).$(fileType)")
  save(savename, figure)

  #figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, title="ANN with inputs from reference solution, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  #savename = joinpath(plotdir, "surrWithRefInputs_in_prox_$(proximity).$(fileType)")
  #save(savename, figure)
end
