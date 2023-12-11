using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using ChainRulesCore
using Zygote
using BSON
using Flux
using LinearAlgebra
using FMI
using FMIImport

using Statistics
using Plots
using Metrics

import DataFrames
import CSV
import InvertedIndices
import StatsBase


import FMICore
using Libdl

include("clusterData.jl")


function readData(filename::String, nInputs::Integer; ratio=0.9, shuffle::Bool=true)
    df = DataFrames.select(CSV.read(filename, DataFrames.DataFrame; ntasks=1), InvertedIndices.Not([:Trace]))
    m = Matrix{Float32}(df)
    n = length(m[:,1]) # num samples
    num_train = Integer(round(n*ratio))
    if shuffle
      trainIters = StatsBase.sample(1:n, num_train, replace = false)
    else
      trainIters = 1:num_train
    end
    testIters = setdiff(1:n, trainIters)

    train_in  = [m[i, 1:nInputs]     for i in trainIters]
    train_out = [m[i, nInputs+1:end] for i in trainIters]
    test_in   = [m[i, 1:nInputs]     for i in testIters]
    test_out  = [m[i, nInputs+1:end] for i in testIters]
    
    train_in = mapreduce(permutedims, vcat, train_in)'
    train_out = mapreduce(permutedims, vcat, train_out)'
    test_in = mapreduce(permutedims, vcat, test_in)'
    test_out = mapreduce(permutedims, vcat, test_out)'
    return train_in, train_out, test_in, test_out
end



#-------------------------------
# when using residual loss, load fmu
#(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, eq_num, rand(Float64, 110))
#"/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/IEEE_14_Buses.interface.fmu"
#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu
#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson






function prepare_fmu(fmu_path, prof_info_path)
  """
  loads fmu from path
  loads profilinginfo from path
  creates value references for the iteration variables and using variables
  is called once for Initialization
  """
  fmu = FMI.fmiLoad(fmu_path)
  comp = FMI.fmiInstantiate!(fmu)
  FMI.fmiSetupExperiment(comp)
  FMI.fmiEnterInitializationMode(comp)
  FMI.fmiExitInitializationMode(comp)

  profilinginfo = getProfilingInfo(prof_info_path)

  vr = FMI.fmiStringToValueReference(fmu, profilinginfo[1].iterationVariables)

  eq_num = profilinginfo[1].eqInfo.id

  row_value_reference = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].usingVars)

  return comp, fmu, profilinginfo, vr, row_value_reference, eq_num
end

comp, fmu, profilinginfo, vr, row_value_reference, eq_num = prepare_fmu("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson")

                                                          
function prepare_x(x, row_vr, fmu, transform)
  """
  calls SetReal for a model input 
  is called before the forward pass
  (should work for batchsize>1)
  """
  batchsize = size(x)[2]
  if batchsize>1
    for i in 1:batchsize
      x_i = x[1:end,i]
      x_i_rec = StatsBase.reconstruct(transform, x_i)
      FMIImport.fmi2SetReal(fmu, row_vr, x_i_rec)
    end
  else
    x_rec = StatsBase.reconstruct(transform, x)
    FMIImport.fmi2SetReal(fmu, row_vr, vec(x_rec))
  end
#   if time !== nothing
#     FMIImport.fmi2SetTime(fmu, x[1])
#     FMIImport.fmi2SetReal(fmu, row_vr, x[2:end])
#   else
#     FMIImport.fmi2SetReal(fmu, row_vr, x[1:end])
#   end
end

b = -0.5
function compute_x_from_y(s, r, y)
  return (r*s+b)-y
end

function compute_residual(y, fmu, eq_num)
  _,res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y))
  return res
end

#BATCH
function loss(y_hat, fmu, eq_num, transform)
  # transform is either train_out_transform or test_out_transform
  """
  y_hat is model output
  evaluates residual of system eq_num at y_hat
  if y_hat is close to a solution, residual is close to 0
  actual loss is the norm of the residual
  """
  batchsize = size(y_hat)[2]
  if batchsize>1
    residuals = []
    for i in 1:batchsize
      y_hat_i = y_hat[1:end,i]
      y_hat_i_rec = StatsBase.reconstruct(transform, y_hat_i)
      _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat_i_rec))
      push!(residuals, res_out)
    end
    return mean(LinearAlgebra.norm.(residuals)), residuals
  else
    y_hat_rec = StatsBase.reconstruct(transform, y_hat)
    _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat_rec))
    return LinearAlgebra.norm(res_out), res_out
  end
  # p = 0
  # if y_hat_rec[1] < 0
  #   p = abs(y_hat_rec[1])
  # end
end


# rrule for loss(x,y)
#BATCH
function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, transform)
  """
  reverse rule for loss function
  needs the jacobian of the system eq_num evaluated at x (x is model output)
  uses that formula: https://math.stackexchange.com/questions/291318/derivative-of-the-2-norm-of-a-multivariate-function
  """
  l, res_out = loss(x, fmu, eq_num, transform) # res_out: residual output, what shape is that?
  batchsize = size(x)[2]
  print(batchsize)
  if batchsize>1
    jacobians = []
    for i in 1:batchsize
      x_i = x[1:end,i]
      x_i_rec = StatsBase.reconstruct(transform, x_i)
      _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x_i_rec))
      mat_dim = trunc(Int,sqrt(length(jac)))
      jac = reshape(jac, (mat_dim,mat_dim))
      push!(jacobians, jac)
    end
  else
    x_rec = StatsBase.reconstruct(train_out_transform, x)
    _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x_rec))
    mat_dim = trunc(Int,sqrt(length(jac)))
    jac = reshape(jac, (mat_dim,mat_dim))
  end

  function loss_pullback(l̄)
    l_tangent = l̄[1] # upstream gradient
    f̄ = NoTangent()
    if batchsize>1
      x̄ = []
      for i in 1:batchsize
        x̄_i = l_tangent * ((jacobians[i]' * res_out[i]) / l) # <-------------- ACTUAL derivative, result should be of shape (110,)
        push!(x̄, x̄_i)
      end
      x̄ = reshape(hcat(x̄...), size(x̄[1])[1], size(x̄)[1])
      print(x̄)
    else
      x̄ = l_tangent * ((jac' * res_out) / l) # <-------------- ACTUAL derivative, result should be of shape (110,)
    end
    # res_out[2] (110,) jac' (110,110) jac'*res_out[2] (110,) x̄ (110,)
    fmū = NoTangent()
    eq_num̄ = NoTangent()
    transform̄ = NoTangent()
    return (f̄, x̄, fmū, eq_num̄, transform̄)
  end
  #jac = rand(110,110) # jacobian dresidual/dx, still need that, probably of shape (110x16), what shape is that?
  # IST DIE QUADRATISCH?

  # x̄ (110,) so wie model output
  return l, loss_pullback
end


fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/data/eq_14.csv"
nInputs = 2
nOutputs = 1

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

CLUSTER = false
if CLUSTER
  x = compute_x_from_y.(train_in[1,:], train_in[2,:], train_out[1,:])
  train_out = hcat(x, train_out')'

  x = compute_x_from_y.(test_in[1,:], test_in[2,:], test_out[1,:])
  test_out = hcat(x, test_out')'


  cluster_indices, num_clusters = cluster_data(train_out)

  cluster_index = 2
  train_in = extract_cluster(train_in, cluster_indices, cluster_index)
  train_out = extract_cluster(train_out, cluster_indices, cluster_index)

  train_out = train_out[2,:]

  cluster_indices_test, num_clusters_test = cluster_data(test_out)
  cluster_index_test = 2
  test_in = extract_cluster(test_in, cluster_indices_test, cluster_index_test)
  test_out = extract_cluster(test_out, cluster_indices_test, cluster_index_test)

  test_out = test_out[2,:]


  # scale data between [0,1]
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=1)

  train_in = StatsBase.transform(train_in_transform, train_in)
  train_out = StatsBase.transform(train_out_transform, train_out)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=1)

  test_in = StatsBase.transform(train_in_transform, test_in)
  test_out = StatsBase.transform(train_out_transform, test_out)

else
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=2)

  train_in = StatsBase.transform(train_in_transform, train_in)
  train_out = StatsBase.transform(train_out_transform, train_out)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=2)

  test_in = StatsBase.transform(test_in_transform, test_in)
  test_out = StatsBase.transform(test_out_transform, test_out)
end


# only batchsize=1!!
dataloader = Flux.DataLoader((train_in, train_out), batchsize=8, shuffle=true)

# specify network architecture
# maybe add normalization layer at Start
hidden_width = 100
model = Flux.Chain(
  Flux.Dense(nInputs, hidden_width, sigmoid),
  Flux.Dense(hidden_width, nOutputs)
)

ps = Flux.params(model)
opt = Flux.Adam(1e-3)
opt_state = Flux.setup(opt, model)

test_loss_vector = []

function compute_test_loss(model, test_in)
  m = 0
  for i in axes(test_in,2)
      prepare_x(test_in[:,i], row_value_reference, fmu, test_in_transform)
      m += loss(model(test_in[:,i]), fmu, eq_num, test_out_transform)[2][1]^2
  end
  return m / size(test_in,2)
end


# problem: geht nur mit batchsize = 1
num_epochs = 10
epoch_range = 1:num_epochs
for epoch in epoch_range
    for (x, y) in dataloader
        prepare_x(x, row_value_reference, fmu, train_in_transform)
        lv, grads = Flux.withgradient(model) do m  
          prediction = m(x)
          # different losses
          #1.0 * Flux.mse(prediction, y) + 1.0 * loss(prediction, fmu, eq_num)
          #Flux.mse(prediction, y)
          loss(prediction, fmu, eq_num, train_out_transform)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    push!(test_loss_vector, compute_test_loss(model, test_in))
end

# plot loss curve
function plot_curve(curve; kwargs...)
  x = 1:length(curve)
  y = curve
  plot(x, y; kwargs...)
end

plot_curve(test_loss_vector; title="test loss")




# plot x and y
scatter(compute_x_from_y.(test_in[1,:],test_in[2,:],vec(test_out)), vec(test_out))
p = model(test_in)
scatter!(compute_x_from_y.(test_in[1,:],test_in[2,:],vec(p)), vec(p))



# supervised works in batchsize>1
# residual doesnt need batchsize=1 in dataloader and




Flux.mse(model(train_in), train_out)
# MSE auf TRAININGSDATEN
# mse nach 100 epochs mit mse loss: 0.05
# mse nach 100 epochs mit residual loss: 0.24
# mse nach 100 epochs mit combined loss: 0.16

Flux.mse(model(test_in), test_out)
# MSE auf TESTDATEN
# mse nach 100 epochs mit mse loss: 0.06
# mse nach 100 epochs mit residual loss: 0.59
# mse nach 100 epochs mit combined loss: 0.33



#TODO:
# think about ways to cluster output data if not one dimensional
# -> make it two dimensional, try it out
# -> https://stackoverflow.com/questions/11513484/1d-number-array-clustering

# think about different losses or other ways to use the residual/jacobian
# -> regularization

# when should you cluster?
# -> cluster then split or split then cluster?

print("fjfjfjfjf")