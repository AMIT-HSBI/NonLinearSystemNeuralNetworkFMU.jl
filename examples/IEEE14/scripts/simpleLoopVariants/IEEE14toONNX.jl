# train unsupervised model on IEEE14 data and save that model to ONNX using ONNXNaiveNASflux.jl

# imports
using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using ChainRulesCore
using Flux
using LinearAlgebra
using Statistics
using Plots

using ONNXNaiveNASflux

include("utils.jl");
include("trainUnsupervised.jl");


# data preparation
# IEEE14 residual
# TODO: generate more data
data_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/data/eq_1403.csv";
nInputs = 16;
nOutputs = 110;

interface_fmu_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/IEEE_14_Buses.interface.fmu";
profinfo_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/profilingInfo.bson";
modelfile_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1500/temp-profiling/IEEE_14_Buses.c";

comp, fmu, profilinginfo, vr, row_value_reference, eq_num, sys_num = prepare_fmu(interface_fmu_path, profinfo_path, modelfile_path);

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(data_path, nInputs);

in_data = hcat(train_in, test_in)
out_data = hcat(train_out, test_out)


# TODO: maybe remove this problem
in_data = in_data[2:end,:];  # take time variable out of in data

num_samples = 1500 # max is 1500
in_data = in_data[:, 1:num_samples]
out_data = out_data[:, 1:num_samples]


# split data into train and test
train_in, train_out, test_in, test_out = split_train_test(in_data, out_data);
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out);
dataloader = Flux.DataLoader((train_in, train_out), batchsize=1, shuffle=true);

# model initilization
hidden_width = 100;
model = Flux.Chain(
  Flux.Dense(nInputs-1, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, hidden_width, relu),
  Flux.Dense(hidden_width, nOutputs)
);

opt = Flux.Adam(1e-4);
epochs = 500;

# model training
unsupervised_model, unsupervised_test_loss_hist, res_unsup, res_train_unsup, unsupervised_time = trainModelUnsupervised(
    deepcopy(model), deepcopy(opt), dataloader, test_in, test_out, train_in_t, test_in_t, train_out_t, test_out_t, eq_num, sys_num, deepcopy(row_value_reference), deepcopy(fmu); epochs=epochs
    );


# results
using Statistics
plot_loss_history(unsupervised_test_loss_hist; label="unsupervised MSE")
plot_loss_history(res_unsup; label="unsupervised Residual test")
plot_loss_history!(res_train_unsup; label="unsupervised Residual train")
println("mean train res over last 100 epochs ", mean(res_train_unsup[Int(epochs/2):end]))
println("mean test res over last 100 epochs ", mean(res_unsup[Int(epochs/2):end]))
println(unsupervised_time/60, " minutes")

plot_loss_history(res_train_unsup[Int(epochs/2):end])

# model saving and loading
function saveModelToOnnx(filename::String, model, numinputs::Int)
    ONNXNaiveNASflux.save(filename, model, (numinputs, 1))
end

function loadModelFromOnnx(filename::String, numinputs::Int)
    return ONNXNaiveNASflux.load(filename, (numinputs, 1))
end

# save trained model to onnx
onnxname = "ieee14unsupervised2.onnx"
saveModelToOnnx(onnxname, unsupervised_model, nInputs-1)

# load trained model from onnx
modelgraph = loadModelFromOnnx(onnxname, nInputs-1)

# inference on loaded model

random_input = rand(nInputs-1, 2)
unsupervised_model(random_input) ≈ modelgraph(random_input)