using BSON
using CairoMakie
using CSV
using DataFrames
using Flux
using NonLinearSystemNeuralNetworkFMU

include("trainFlux.jl")

"""
Calculate loss of the surrogate on train data
"""
function calcLoss(modelName, N)
  workdir = datadir("sims", "$(modelName)_$(N)")

  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  prof = Array{ProfilingInfo}(dict[first(keys(dict))])[1]

  trainData_csv = datadir("exp_pro", "eq_$(prof.eqInfo.id)_$(N)_proximity.csv")
  trainData = CSV.read(trainData_csv, DataFrame; ntasks=1)

  trainLoss = DataFrames.DataFrame(loss=zeros(size(trainData, 1)))
  BSON.@load joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx.bson") model

  for (i, _) in enumerate(eachrow(trainData))
    usingVars = trainData[i, prof.usingVars]
    oldVars = trainData[i, prof.iterationVariables.*"_old"]
    solution = trainData[i, prof.iterationVariables]
    input = vcat(Array{Float32}(usingVars), Array{Float32}(oldVars))
    trainLoss[i, :loss] = sqrt(sum(abs2.(model(input) .- Array{Float32}(solution))))
  end

  trainLoss_csv = joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id)_loss.csv")
  CSV.write(trainLoss_csv, trainLoss)
end

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
  trainData_csv = joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv")
  trainLoss_csv = joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id)_loss.csv")

  ref_results = CSV.read(ref_csv, DataFrame; ntasks=1)
  onnx_results = CSV.read(onnx_csv, DataFrame; ntasks=1)
  trainData = CSV.read(trainData_csv, DataFrame; ntasks=1)
  trainLoss = CSV.read(trainLoss_csv, DataFrame; ntasks=1)
  trainData.loss = trainLoss.loss

  figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, df_trainData=trainData, title="Training Area", epsilon=0.1)
  display(figure)
  savename = joinpath(plotdir, "trainArea.$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, df_trainData=trainData, title="Training Area t∈[0,0.1]", epsilon=0.1, tspan=(0.0, 0.1))
  savename = joinpath(plotdir, "trainAreaStart.$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.iterationVariables, ref_results; df_surrogate=onnx_results, title="Surrogate Results", epsilon=0.1)
  savename = joinpath(plotdir, "surrogateSolution.$(fileType)")
  save(savename, figure)
end

"""
Plot the solution of the ANN surrogate when using the reference solution as input,
so errors wont influence the next time step.
"""
function SurrogateRefRes(modelName, N; fileType="svg", proximity=1)
  #@assert Threads.nthreads() == 1 "This breaks with multiple threads!"

  workdir = datadir("sims", "$(modelName)_$(N)")
  plotdir = plotsdir("$(modelName)_$(N)")
  mkpath(plotdir)

  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  prof = Array{ProfilingInfo}(dict[first(keys(dict))])[1]

  ref_csv = joinpath(workdir, "temp-omsimulator", "$(modelName)_ref.csv")
  ref_results = CSV.read(ref_csv, DataFrame; ntasks=1)

  BSON.@load joinpath(workdir, "onnx", "eq_$(prof.eqInfo.id).onnx.bson") model

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

  csvFile = datadir("exp_pro", "onnxRef", "eq_$(prof.eqInfo.id)_$(N)_p_$(proximity).csv")
  mkpath(dirname(csvFile))
  CSV.write(csvFile, onnx_results)

  t0 = ref_results[proximity, "time"]

  figure = plotTrainArea(prof.iterationVariables, ref_results; df_surrogate=onnx_results, title="ANN with inputs from ref sol, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  savename = joinpath(plotdir, "surrWithRefInputs_out_prox_$(proximity).$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.iterationVariables[2:5], ref_results; df_surrogate=onnx_results, title="ANN with inputs from reference solution, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  savename = joinpath(plotdir, "surrWithRefInputs_4_prox_$(proximity).$(fileType)")
  save(savename, figure)

  figure = plotTrainArea(prof.iterationVariables[2:5], ref_results; df_surrogate=onnx_results, title="ANN with inputs from reference solution, p=$(proximity)", epsilon=0.1, tspan=(0.8, 1.2))
  savename = joinpath(plotdir, "surrWithRefInputs_event_prox_$(proximity).$(fileType)")
  save(savename, figure)

  #figure = plotTrainArea(prof.usingVars, ref_results; df_surrogate=onnx_results, title="ANN with inputs from reference solution, p=$(proximity)", epsilon=0.1, tspan=(t0, 0.98))
  #savename = joinpath(plotdir, "surrWithRefInputs_in_prox_$(proximity).$(fileType)")
  #save(savename, figure)
end
