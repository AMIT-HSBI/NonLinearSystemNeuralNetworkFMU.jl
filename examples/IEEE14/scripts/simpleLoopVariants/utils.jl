import StatsBase
import Clustering
import Distances
import DataFrames
import InvertedIndices
import CSV
import FMI
import Random
using FMIImport
import ChainRulesCore


function loss(y_hat, fmu, eq_num, sys_num, transform)
    bs = size(y_hat)[2] # batchsize
    residuals = Array{Vector{Float64}}(undef, bs)
    for j in 1:bs
        yj_hat = StatsBase.reconstruct(transform, y_hat[:,j])
        _, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(yj_hat))
        residuals[j] = res
    end
    return 1/(2*bs)*sum(norm.(residuals).^2), residuals
end
  
function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform)
    l, res = loss(x, fmu, eq_num, sys_num, transform)
    # evaluate the jacobian for each batch element
    bs = size(x)[2] # batchsize
    res_dim = length(res[1])
    jac_dim = res_dim

    jacobians = Array{Matrix{Float64}}(undef, bs)
    for j in 1:bs
        xj = StatsBase.reconstruct(transform, x[:,j])
        _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(xj))
        jacobians[j] = reshape(jac, (jac_dim,jac_dim))
    end

    function loss_pullback(l̄)
        factor = l̄./bs

        x̄ = Array{Float64}(undef, res_dim, bs)
        # compute x̄
        for j in 1:bs
            x̄[:,j] = transpose(jacobians[j]) * res[j]
        end
        x̄ = if transform.dims == 1 x̄ .* (1 ./ transform.scale)' elseif transform.dims == 2 x̄ .* (1 ./ transform.scale) end
        x̄ .*= factor

        # all other args have NoTangent
        f̄ = ChainRulesCore.NoTangent()
        fmū = ChainRulesCore.NoTangent()
        eq_num̄ = ChainRulesCore.NoTangent()
        sys_num̄ = ChainRulesCore.NoTangent()
        transform̄ = ChainRulesCore.NoTangent()
        return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
    end

    return l, loss_pullback
end


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


function scale_data_uniform(train_in, train_out, test_in, test_out)
    train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
    train_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_out, dims=2)
  
    train_in = StatsBase.transform(train_in_transform, train_in)
    train_out = StatsBase.transform(train_out_transform, train_out)
  
    test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
    test_out_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_out, dims=2)
  
    test_in = StatsBase.transform(train_in_transform, test_in)
    test_out = StatsBase.transform(train_out_transform, test_out)
  
    return train_in, train_out, test_in, test_out, train_in_transform, train_out_transform, test_in_transform, test_out_transform
end

function vectorofvector_to_matrix(vov)
    return mapreduce(permutedims, vcat, vov)
end

function get_cluster_indices(cluster_assignments::Vector{Int})
    # Create a dictionary to store indices for each cluster
    cluster_indices_dict = Dict{Int, Vector{Int}}()

    for (i, cluster) in enumerate(cluster_assignments)
        if haskey(cluster_indices_dict, cluster)
            push!(cluster_indices_dict[cluster], i)
        else
            cluster_indices_dict[cluster] = [i]
        end
    end

    # Convert the dictionary to a list of indices for each cluster
    cluster_indices_list = [cluster_indices_dict[i] for i in 1:maximum(cluster_assignments)]

    return cluster_indices_list
end

function extract_cluster(data, cluster_indices::Vector{Vector{Int64}}, cluster_index::Int64)
    # data is in the form (d, n)
    # d - feature dimension
    # n - number of datapoints
    # cluster_indices: index of clusters
    # cluster_index: index of cluster to get
    return data[:, cluster_indices[cluster_index]]
end

function cluster_data(train_out)
    # train_out is a Matrix of shape n_targetsXn_samples
    train_out_c = copy(train_out)
    dt = StatsBase.fit(StatsBase.ZScoreTransform, train_out_c, dims=1) # normalise along columns
    train_out_c = StatsBase.transform(dt, train_out_c)


    max_score = 0
    max_score_num_cluster = 1
    max_cluster = 20
    distances = Distances.pairwise(Distances.SqEuclidean(), train_out_c)
    for i = 2:max_cluster
        R = Clustering.kmeans(train_out_c, i; maxiter=200)
        score = mean(Clustering.silhouettes(R, distances))
        if score > max_score
            max_score = score
            max_score_num_cluster = i
        end
    end

    R = Clustering.kmeans(train_out_c, max_score_num_cluster; maxiter=200)
    cluster_indices = get_cluster_indices(R.assignments)
    return cluster_indices, max_score_num_cluster
end

function scale_data_uniform(train_in, test_in)
    train_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, train_in, dims=2)
    train_in = StatsBase.transform(train_in_transform, train_in)
  
    test_in_transform = StatsBase.fit(StatsBase.UnitRangeTransform, test_in, dims=2)
    test_in = StatsBase.transform(test_in_transform, test_in)
  
    return train_in, test_in, train_in_transform, test_in_transform
end

function parse_modelfile(modelfile_path, eq_num)
  conc_string = "equation index: " * string(eq_num)
  nonlin_string = "indexNonlinear: "
  regex_expression = r"(?<=indexNonlinear: )(\d+)"
  open(modelfile_path) do f
      # line_number
      found = false
      # read till end of file
      while !eof(f) 
          # read a new / next line for every iteration		 
          s = readline(f)

          if found && occursin(nonlin_string, s)
              line_matches = match(regex_expression, s)
              return parse(Int64, line_matches.match)
          end

          if occursin(conc_string, s)
              # read the next line
              found = true
          end
      end
      println("not found")
      return nothing
  end
end


function prepare_fmu(fmu_path, prof_info_path, model_path)
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
  
    inp_value_reference = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].usingVars)
    out_value_reference = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].iterationVariables)
    
    eq_num = profilinginfo[1].eqInfo.id
    sys_num = parse_modelfile(model_path, eq_num)
  
  
    return comp, fmu, profilinginfo, out_value_reference, inp_value_reference, eq_num, sys_num
end


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
            FMIImport.fmi2SetReal(fmu, row_vr, Float64.(x_i_rec))
        end
    else
        x_rec = StatsBase.reconstruct(transform, x)
        FMIImport.fmi2SetReal(fmu, row_vr, Float64.(vec(x_rec)))
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


# plot x and y
function plot_xy(model, in_data, out_data, in_t, out_t; kwargs...)
    prediction = model(in_data)
    in_data_rec = StatsBase.reconstruct(in_t, in_data)
    prediction_rec = StatsBase.reconstruct(out_t, prediction)
    scatter!(compute_x_from_y.(in_data_rec[1,:],in_data_rec[2,:],vec(prediction_rec)), vec(prediction_rec); kwargs...)
end


function plot_loss_history(loss_history, x=nothing; kwargs...)
    if isnothing(x)
        x = 1:length(loss_history)
    end    
    plot(x, loss_history; kwargs...)
  end

  
function plot_loss_history!(loss_history, x=nothing; kwargs...)
    if isnothing(x)
        x = 1:length(loss_history)
    end  
    plot!(x, loss_history; kwargs...)
end


function split_train_test(in_data, out_data, test_ratio=0.2, random_seed=42)
    Random.seed!(random_seed)

    num_samples = size(in_data, 2)
    indices = Random.shuffle(1:num_samples)

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