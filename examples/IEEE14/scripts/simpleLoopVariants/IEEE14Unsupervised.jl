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
using MultivariateStats

include("utils.jl")
include("trainUnsupervised.jl")
include("trainSupervised.jl")
include("trainSemiSupervised.jl")
include("trainTwoStep.jl")

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


CLUSTER = false
if CLUSTER
  # concat in and out data
  in_data = hcat(train_in, test_in)
  out_data = hcat(train_out, test_out)

  # cluster out data
  cluster_indices, num_clusters = cluster_data(out_data)
  # extract cluster
  cluster_index = 1 #rand(1:num_clusters)
  in_data = extract_cluster(in_data, cluster_indices, cluster_index)
  out_data = extract_cluster(out_data, cluster_indices, cluster_index)
else
  # take time out, probably earlier
  in_data = hcat(train_in, test_in)
  out_data = hcat(train_out, test_out)
end

# Warning: No variable named 'time' found.
in_data = in_data[2:end,:] # take time variable out of in data
# and out of the row_value_reference
rvr = []
for i in 2:16
    push!(rvr, row_value_reference[i])
end
rvr = Int64.(rvr)


# split one cluster into train and test
train_in, train_out, test_in, test_out = split_train_test(in_data, out_data)
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
dataloader = Flux.DataLoader((train_in, train_out), batchsize=8, shuffle=true)



hidden_width = 100
model = Flux.Chain(
  Flux.Dense(nInputs-1, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
)
opt = Flux.Adam(1e-4)
#opt = Flux.Optimise.Optimiser(Flux.Adam(1e-4), ExpDecay())
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
                                                                                                       deepcopy(fmu); epochs=100)

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
                                                                                              deepcopy(fmu); epochs=100)

semisupervised_model, semisupervised_test_loss_hist, semisupervised_time = trainModelSemisupervised(deepcopy(model), 
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
                                                                                              deepcopy(fmu); epochs=100, h1=0.8, h2=0.2)

twost_model, twost_test_loss_hist, twost_time = trainTwoStep(deepcopy(model), 
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
                                                            deepcopy(fmu); epochs=100)

                                          

# pca of test out data to 3 dimensions
M1 = fit(PCA, test_out; maxoutdim=3)
O1 = predict(M1, test_out)
scatter(O1[1,:],O1[2,:],O1[3,:], label="groundtruth")

# pca of unsupervised test prediction to 3 dimensions
pred2 = unsupervised_model(test_in)
O2 = predict(M1, pred2)
scatter!(O2[1,:],O2[2,:],O2[3,:], label="unsupervised")

# pca of supervised test prediction to 3 dimensions
pred3 = supervised_model(test_in)
O3 = predict(M1, pred3)
scatter!(O3[1,:],O3[2,:],O3[3,:], label="supervised")

pred4 = semisupervised_model(test_in)
O4 = predict(M1, pred4)
scatter!(O4[1,:],O4[2,:],O4[3,:], label="semi-supervised")

pred5 = twost_model(test_in)
O5 = predict(M1, pred5)
scatter!(O5[1,:],O5[2,:],O5[3,:], label="twostep")


plot_loss_history(unsupervised_test_loss_hist; label="unsupervised")
plot_loss_history!(supervised_test_loss_hist; label="supervised")
plot_loss_history!(semisupervised_test_loss_hist; label="semi-supervised")
plot_loss_history!(twost_test_loss_hist; label="twostep")
title!("MSE test loss")
xlabel!("Number of Epochs")
ylabel!("MSE")


plot_loss_history(res_unsup; label="unsupervised")
plot_loss_history!(res_sup; label="supervised")
title!("Residual test loss")
xlabel!("Number of Epochs")
ylabel!("Residual")


# ⟹ unsupervised doesnt work at all in high dimensions, maybe too many solutions to converge to
# ⟹ the reasons could be: model not powerful enough, training too short (no!)
# ⟹ semi-supervised works pretty good, but maybe because its basically supervised
# ⟹ semi-supervised could work when you have little data from one cluster where you wanna pin the solution and use residual information for the rest

# ¡ check if unsupervised converges to a solution or not using clustering and pca


(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)
# concat in and out data
in_data = hcat(train_in, test_in)
out_data = hcat(train_out, test_out)

# cluster out data
cluster_indices, num_clusters = cluster_data(out_data)
train_in, train_out, test_in, test_out = split_train_test(in_data, out_data)
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
M = fit(PCA, train_out; maxoutdim=3)


for i in 1:num_clusters
  in_cluster = extract_cluster(in_data, cluster_indices, i)
  out_cluster = extract_cluster(out_data, cluster_indices, i)
  p = predict(M, out_cluster)
  scatter!(pl, p[1,:],p[2,:],p[3,:], label="cluster $i")
end


display(pl)

# ideas:
# - two step training procedure, doesnt work naive, maybe freeze some layers after step 1
# - tuning of semi supervised loss function while training






mm = Flux.Chain(
  Flux.Dense(15, 30, relu),
  Flux.Dense(30, 30, relu),
  Flux.Dense(30, 110)
)

op_chain = Flux.Adam(1e-4)
opt = Flux.Optimisers.setup(op_chain, mm)

Flux.Optimisers.freeze!(opt.layers[1])

for (x, y) in dataloader
  lv, grads = Flux.withgradient(mm) do m  
    prediction = m(x)
    loss(prediction, fmu, eq_num, sys_num, train_out_t)
  end
  Flux.Optimisers.update(opt, mm, grads[1])
end



M = fit(PCA, test_out; maxoutdim=3)
O1 = predict(M1, test_out)

pl = plot()
scatter!(pl, O1[1,:],O1[2,:],O1[3,:], label="groundtruth")
hp = [0.0, 0.25, 0.5, 0.75, 1.0] # amount of residual in semisupervised setting
for h in hp
  semisupervised_model, semisupervised_test_loss_hist, semisupervised_time = trainModelSemisupervised(deepcopy(model), 
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
                                                                                              deepcopy(fmu); epochs=100, h1=1-h, h2=h)
    pred_ = semisupervised_model(test_in)
    O4 = predict(M, pred_)
    scatter!(pl, O4[1,:],O4[2,:],O4[3,:], label="h = $h")
end
display(pl)
