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

include("utils.jl")
include("trainUnsupervised.jl")
include("trainSupervised.jl")

# prepare the data for multiple experiments
fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/data/eq_1403.csv"
nInputs = 16
nOutputs = 110

comp, fmu, profilinginfo, vr, row_value_reference, eq_num, sys_num = prepare_fmu(
  "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/IEEE_14_Buses.interface.fmu",
  "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/profilingInfo.bson",
  "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/temp-profiling/IEEE_14_Buses.c")

# 1. some UNAMBIGOUS dataset and train all methods for performance
# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)


# # concat in and out data
# in_data = hcat(train_in, test_in)
# out_data = hcat(train_out, test_out)

# # cluster out data
# cluster_indices, num_clusters = cluster_data(out_data)
# # extract cluster
# cluster_index = 1 #rand(1:num_clusters)
# in_data = extract_cluster(in_data, cluster_indices, cluster_index)
# out_data = extract_cluster(out_data, cluster_indices, cluster_index)

# take time out, probably earlier
in_data = hcat(train_in, test_in)
out_data = hcat(train_out, test_out)
in_data = in_data[2:end,:]
rvr = []
for i in 2:16
    push!(rvr, row_value_reference[i])
end
rvr = Int64.(rvr)


# split one cluster into train and test
train_in, train_out, test_in, test_out = split_train_test(in_data, out_data)
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
dataloader = Flux.DataLoader((train_in, train_out), batchsize=32, shuffle=true)


struct prelu{Float64}
  alpha::Float64
  function prelu(alpha_init::Float64 = 0.25)
      new{Float64}(alpha_init)
  end
end

function (a::prelu)(x::AbstractArray)
  pos = Flux.relu(x)
  neg = -a.alpha * Flux.relu(-x)
  return pos + neg
end

Flux.@functor prelu

hidden_width = 150
model = Flux.Chain(
  Flux.Dense(nInputs-1, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
)
#opt = Flux.Adam(1e-4)
opt = Flux.Optimise.Optimiser(Flux.Adam(1e-4), ExpDecay())
#state = Optimisers.setup(Optimisers.Adam(), model)


unsupervised_model, unsupervised_test_loss_hist, res_unsup, unsupervised_time = trainModelUnsupervised(deepcopy(model), 
                                                                                                       deepcopy(opt), 
                                                                                                       dataloader, 
                                                                                                       test_in, 
                                                                                                       test_out, 
                                                                                                       train_in_t, 
                                                                                                       test_in_t, 
                                                                                                       train_out_t, 
                                                                                                       test_out_t, 
                                                                                                       eq_num, 
                                                                                                       sys_num, 
                                                                                                       deepcopy(rvr), 
                                                                                                       deepcopy(fmu); epochs=1000)

supervised_model, supervised_test_loss_hist, res_sup, supervised_time = trainModelSupervised(deepcopy(model), 
                                                                                              deepcopy(opt), 
                                                                                              dataloader, 
                                                                                              test_in, 
                                                                                              test_out, 
                                                                                              train_in_t, 
                                                                                              test_in_t, 
                                                                                              train_out_t, 
                                                                                              test_out_t, 
                                                                                              eq_num, 
                                                                                              sys_num, 
                                                                                              deepcopy(rvr), 
                                                                                              deepcopy(fmu); epochs=1000)

                                              


plot_loss_history(unsupervised_test_loss_hist)
plot_loss_history(supervised_test_loss_hist)

plot_loss_history(res_unsup)
plot_loss_history(res_sup)



mm = Flux.Chain(
  Flux.Dense(15, 30, relu),
  Flux.Dense(30, 30, relu),
  Flux.Dense(30, 110)
)

op_chain = Flux.Optimisers.OptimiserChain(ExpDecay(), Flux.Adam(1e-4))
opt = Flux.Optimisers.setup(op_chain, mm)

for (x, y) in dataloader
  lv, grads = Flux.withgradient(mm) do m  
    prediction = m(x)
    loss(prediction, fmu, eq_num, sys_num, train_out_t)
  end
  Flux.Optimisers.update(opt, mm, grads[1])
end