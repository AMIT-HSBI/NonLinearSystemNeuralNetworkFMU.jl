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
include("trainSupervised.jl")
include("trainUnsupervised.jl")
include("trainSemiSupervised.jl")


# prepare the data for multiple experiments
fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/data/eq_14.csv"
nInputs = 2
nOutputs = 1

comp, fmu, profilinginfo, vr, row_value_reference, eq_num, sys_num = prepare_fmu("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/temp-profiling/simpleLoop.c")

# 1. some UNAMBIGOUS dataset and train all methods for performance
# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)
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
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
dataloader = Flux.DataLoader((train_in, train_out), batchsize=1, shuffle=true)

hidden_width = 100
model = Flux.Chain(
  Flux.Dense(nInputs, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
)
opt = Flux.Adam(1e-4)

supervised_model, supervised_test_loss_hist, supervised_time = trainModelSupervised(deepcopy(model), deepcopy(opt), dataloader, test_in, test_out;epochs=1000)
unsupervised_model, unsupervised_test_loss_hist, unsupervised_time = trainModelUnsupervised(deepcopy(model), deepcopy(opt), train_in, test_in, train_in_t, test_in_t, train_out_t, test_out_t, eq_num, sys_num, row_value_reference, fmu; test_out=test_out, epochs=1000)
semisupervised_model, semisupervised_test_loss_hist, semisupervised_time = trainModelSemisupervised(deepcopy(model), deepcopy(opt), train_in, test_in, train_in_t, test_in_t, train_out_t, test_out_t, eq_num, sys_num, row_value_reference, fmu; test_out=test_out, epochs=1000)


# plot mse loss results
plot_loss_history(supervised_test_loss_hist; label="supervised")
plot_loss_history!(unsupervised_test_loss_hist; label="unsupervised")
plot_loss_history!(semisupervised_test_loss_hist; label="semi-supervised")
title!("MSE for clustered dataset")
xlabel!("Number of Epochs")
ylabel!("MSE")


# plot xy results
#TODO: think about the correct scaling of data
scatter(compute_x_from_y.(test_in[1,:],test_in[2,:],vec(test_out)), vec(test_out), label="groundtruth")
plot_xy(supervised_model, test_in, test_out; label="supervised")
plot_xy(unsupervised_model, test_in, test_out; label="unsupervised")
plot_xy(semisupervised_model, test_in, test_out; label="semi-supervised")
title!("XY plot for clustered dataset")
xlabel!("x")
ylabel!("y")




# 2. some AMBIGOUS dataset and train all methods for performance
# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
dataloader = Flux.DataLoader((train_in, train_out), batchsize=8, shuffle=true) #???

hidden_width = 100
model = Flux.Chain(
  Flux.Dense(nInputs, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
)
opt = Flux.Adam(1e-4)

supervised_model, supervised_test_loss_hist, supervised_time = trainModelSupervised(deepcopy(model), deepcopy(opt), dataloader, test_in, test_out;epochs=1000)
unsupervised_model, unsupervised_test_loss_hist, unsupervised_time = trainModelUnsupervised(deepcopy(model), deepcopy(opt), train_in, test_in, train_in_t, test_in_t, train_out_t, test_out_t, eq_num, sys_num, row_value_reference, fmu; test_out=test_out, epochs=1000)
semisupervised_model, semisupervised_test_loss_hist, semisupervised_time = trainModelSemisupervised(deepcopy(model), deepcopy(opt), train_in, test_in, train_in_t, test_in_t, train_out_t, test_out_t, eq_num, sys_num, row_value_reference, fmu; test_out=test_out, epochs=1000)


# plot mse loss results
plot_loss_history(supervised_test_loss_hist; label="supervised")
plot_loss_history!(unsupervised_test_loss_hist; label="unsupervised")
plot_loss_history!(semisupervised_test_loss_hist; label="semi-supervised")
title!("MSE for unclustered dataset")
xlabel!("Number of Epochs")
ylabel!("MSE")


# plot xy results
#TODO: think about the correct scaling of data
scatter(compute_x_from_y.(test_in[1,:],test_in[2,:],vec(test_out)), vec(test_out), label="groundtruth")
plot_xy(supervised_model, test_in, test_out; label="supervised")
plot_xy(unsupervised_model, test_in, test_out; label="unsupervised")
plot_xy(semisupervised_model, test_in, test_out; label="semi-supervised")
title!("XY plot for unclustered dataset")
xlabel!("x x=(r*s+b)-y")
ylabel!("y prediction")


# 3. compare training time between all methods when using fully unsupervised training and fully supervised training

# 4. ideas and testing to improve model performance (lr decay, regularization, prelu, dropout, batch norm)


#TODO
# track residual loss
# plot training time
