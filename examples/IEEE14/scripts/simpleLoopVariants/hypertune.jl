# adapted from: https://github.com/FluxML/model-zoo

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onecold, @epochs, flatten
using Flux.Losses: mse
using Base: @kwdef
using HyperTuning
import Random


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
train_loader = Flux.DataLoader((train_in, train_out), batchsize=1, shuffle=true);



function objective(trial)
    # fix seed for the RNG
    seed = get_seed(trial)
    Random.seed!(seed)

    
    # get suggested hyperparameters
    @suggest activation in trial
    @suggest n_dense in trial
    @suggest dense in trial

    # Create the model with dense layers (fully connected)
    ann = []
    n_input = nInputs-1
    for n in dense[1:n_dense]
        push!(ann, Dense(n_input, n, activation))
        n_input = n
    end
    push!(ann, Dense(n_input, nOutputs))
    model = Chain(ann)
    # model parameters
    #ps = Flux.params(model)  

    # hyperparameters for the optimizer
    @suggest optimizer in trial
    @suggest η in trial
    #@suggest λ in trial

    # Instantiate the optimizer
    #opt = λ > 0 ? Flux.Optimiser(WeightDecay(λ), ADAM(η)) : ADAM(η)
    opt = optimizer(η)
    opt_state = Flux.setup(opt, model)

    epochs = 40 # maximum number of training epochs
    test_loss = 10.0
    # Training
    for epoch in 1:epochs
        for (x, y) in train_loader #!!!!!!!!
            # forward, grad computation
            lv, grads = Flux.withgradient(model) do m  
                prediction = m(x)
                loss(prediction, fmu, eq_num, sys_num, train_out_t)
            end
            Flux.update!(opt_state, model, grads[1])
            #Flux.Optimise.update!(opt, ps, grads[1])
        end

        prepare_x(test_in, row_value_reference, fmu, test_in_t)
        test_loss, _ = loss(model(test_in), fmu, eq_num, sys_num, test_out_t)
        # report value to pruner
        report_value!(trial, test_loss)
        # check if pruning is necessary
        should_prune(trial) && (return)
    end
    
    # if accuracy is over 90%, then trials is considered as feasible
    test_loss < 0.001 && report_success!(trial)
    # return objective function value
    return test_loss
end

# maximum and minimum number of dense layers
const MIN_DENSE = 3
const MAX_DENSE = 7

scenario = Scenario(### hyperparameters
                    # learning rates
                    η = (0.0..0.5),
                    #λ = (0.0..0.5),
                    # optimizer
                    optimizer = [Adam, AdaGrad],
                    # activation functions
                    activation = [relu, elu],
                    # number of dense layers
                    n_dense = MIN_DENSE:MAX_DENSE,
                    # number of neurons for each dense layer
                    dense = Bounds(fill(10, MAX_DENSE), fill(100, MAX_DENSE)),
                    ### Common settings
                    pruner = MedianPruner(start_after = 5#=trials=#, prune_after = 10#=epochs=#),
                    verbose = true, # show the log
                    max_trials = 50, # maximum number of hyperparameters computed
                   )

display(scenario)

# minimize residual error
HyperTuning.optimize(objective, scenario)



# TODO: change to implicit style like in: https://github.com/jmejia8/hypertuning-examples/blob/main/Flux/flux.jl