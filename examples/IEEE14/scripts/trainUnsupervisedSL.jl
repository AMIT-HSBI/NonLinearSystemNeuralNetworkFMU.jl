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


using Random
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
      #y_hat_i_rec = StatsBase.reconstruct(transform, y_hat_i)
      _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat_i))
      push!(residuals, res_out)
    end
    return mean(LinearAlgebra.norm.(residuals)), residuals
  else
    #y_hat_rec = StatsBase.reconstruct(transform, y_hat)
    _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat))
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
  # evaluate the jacobian for each batch element
  batchsize = size(x)[2]
  if batchsize>1
    jacobians = []
    for i in 1:batchsize
      x_i = x[1:end,i]
      _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x_i))
      mat_dim = trunc(Int,sqrt(length(jac)))
      jac = reshape(jac, (mat_dim,mat_dim))
      push!(jacobians, jac)
    end
  else
    _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x[:,1]))
    mat_dim = trunc(Int,sqrt(length(jac)))
    jac = reshape(jac, (mat_dim,mat_dim))
  end

  function loss_pullback(l̄)
    l_tangent = l̄[1] # upstream gradient

    # compute x̄
    if batchsize>1
      # backprop through mean of norms of batch elements
      factor = l_tangent/(batchsize*l)
      x̄ = jacobians[1]' * res_out[1]
      for i in 2:batchsize
        x̄ = x̄ + jacobians[i]' * res_out[i]
      end
      x̄*=factor
      x̄ = repeat(x̄, 1, batchsize)
    else
      x̄ = l_tangent * ((jac' * res_out) / l)
    end

    # all other args have NoTangent
    f̄ = NoTangent()
    fmū = NoTangent()
    eq_num̄ = NoTangent()
    transform̄ = NoTangent()
    return (f̄, x̄, fmū, eq_num̄, transform̄)
  end
  return l, loss_pullback
end


fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/data/eq_14.csv"
nInputs = 2
nOutputs = 1

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

function split_train_test(in_data, out_data, test_ratio=0.2, random_seed=42)
    Random.seed!(random_seed)

    num_samples = size(in_data, 2)
    indices = shuffle(1:num_samples)

    # Calculate the number of samples for the test set
    num_test = round(Int, test_ratio * num_samples)

    # Split the indices into training and testing sets
    train_indices = indices[1:(num_samples - num_test)]
    test_indices = indices[(num_samples - num_test + 1):end]

    # Create training and testing sets
    train_in = in_data[:, train_indices]
    train_out = out_data[:, train_indices]
    test_in = in_data[:, test_indices]
    test_out = out_data[:, test_indices]

    return train_in, train_out, test_in, test_out
end

using Random

function split_train_test(data_matrix, test_ratio=0.2, random_seed=42)
    Random.seed!(random_seed)

    num_samples = size(data_matrix, 2)
    indices = shuffle(1:num_samples)

    # Calculate the number of samples for the test set
    num_test = round(Int, test_ratio * num_samples)

    # Split the indices into training and testing sets
    train_indices = indices[1:(num_samples - num_test)]
    test_indices = indices[(num_samples - num_test + 1):end]

    # Create training and testing sets
    train_data = data_matrix[:, train_indices]
    test_data = data_matrix[:, test_indices]

    return train_data, test_data
end


function extract_using_var_bounds(profilinginfo)
  pf = profilinginfo[1]
  num_using_vars = length(pf.usingVars)
  min_bound = pf.boundary.min
  max_bound = pf.boundary.max
  return min_bound, max_bound, num_using_vars
end


function generate_unsupervised_data(profilinginfo, num_points)
  min_bound, max_bound, num_uv = extract_using_var_bounds(profilinginfo)
  data_matrix = zeros(num_uv, num_points)

  for i in 1:num_uv
    feature_min = min_bound[i]
    feature_max = max_bound[i]
    data_matrix[i, :] .= rand(Float32, num_points) * (feature_max - feature_min) .+ feature_min
  end

  return data_matrix
end

function scale_data_uniform(train_in, train_out, test_in, test_out)
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=2)

  train_in = StatsBase.transform(train_in_transform, train_in)
  train_out = StatsBase.transform(train_out_transform, train_out)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=2)

  test_in = StatsBase.transform(test_in_transform, test_in)
  test_out = StatsBase.transform(test_out_transform, test_out)

  return train_in, train_out, test_in, test_out, train_in_transform, train_out_transform, test_in_transform, test_out_transform
end

function scale_data_uniform(train_in, test_in)
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_in = StatsBase.transform(train_in_transform, train_in)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_in = StatsBase.transform(test_in_transform, test_in)

  return train_in, test_in, train_in_transform, test_in_transform
end


CLUSTER = nothing
if CLUSTER == true
  # concat in and out data
  in_data = hcat(train_in, test_in)
  out_data = hcat(train_out, test_out)

  x = compute_x_from_y.(in_data[1,:], in_data[2,:], out_data[1,:])
  out_data = hcat(x, out_data')'
  # cluster out data
  cluster_indices, num_clusters = cluster_data(out_data)
  # extract cluster
  cluster_index = 1 #rand(1:num_clusters)
  in_data = extract_cluster(in_data, cluster_indices, cluster_index)
  out_data = extract_cluster(out_data, cluster_indices, cluster_index)
  out_data = out_data[2,:]
  out_data = reshape(out_data, 1, length(out_data))
  # split one cluster into train and test
  train_in, train_out, test_in, test_out = split_train_test(in_data, out_data)


  # scale data between [0,1]
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=2)

  train_in = StatsBase.transform(train_in_transform, train_in)
  train_out = StatsBase.transform(train_out_transform, train_out)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=2)

  test_in = StatsBase.transform(train_in_transform, test_in)
  test_out = StatsBase.transform(train_out_transform, test_out)

elseif CLUSTER == false
  train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
  train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=2)

  train_in = StatsBase.transform(train_in_transform, train_in)
  train_out = StatsBase.transform(train_out_transform, train_out)

  test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
  test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=2)

  test_in = StatsBase.transform(test_in_transform, test_in)
  test_out = StatsBase.transform(test_out_transform, test_out)
end



# train fully unsupervised
unsupervised_data_matrix = generate_unsupervised_data(profilinginfo, 1000)
train_data, test_data = split_train_test(unsupervised_data_matrix)
train_data, test_data, train_data_transform, test_data_transform = scale_data_uniform(train_data, test_data)
unsupervised_dataloader = Flux.DataLoader(train_data, batchsize=1, shuffle=true)

scatter(train_data[1,:], train_data[2,:])
scatter!(test_data[1,:], test_data[2,:])

# only batchsize=1!!
dataloader = Flux.DataLoader((train_in, train_out), batchsize=16, shuffle=true)

# specify network architecture
# maybe add normalization layer at Start
hidden_width = 100
model = Flux.Chain(
  Flux.Dense(nInputs, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
)

ps = Flux.params(model)
opt = Flux.Adam(1e-4)
opt_state = Flux.setup(opt, model)

test_loss_residual = []
test_loss_mse = []

function comp_test_loss_residual(model, test_in)
  prepare_x(test_in, row_value_reference, fmu, test_data_transform)
  l,_ = loss(model(test_in), fmu, eq_num, test_out_transform)
  return l
end


num_epochs = 10
epoch_range = 1:num_epochs
for epoch in epoch_range
    for x in unsupervised_dataloader
        prepare_x(x, row_value_reference, fmu, train_data_transform)
        lv, grads = Flux.withgradient(model) do m  
          prediction = m(x)
          # different losses
          #Flux.mse(prediction, y) + 0.2 * (loss(prediction, fmu, eq_num, train_out_transform)/10)
          #Flux.mse(prediction, y)
          loss(prediction, fmu, eq_num, train_out_transform)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    push!(test_loss_residual, comp_test_loss_residual(model, test_data))
    #push!(test_loss_mse, Flux.mse(model(test_in), test_out))
end

# plot loss curve
function plot_curve(curve; kwargs...)
  x = 1:length(curve)
  y = curve
  plot(x, y; kwargs...)
end

plot_curve(test_loss_residual; title="test loss residual")
#plot_curve(test_loss_mse; title="test loss mse")




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