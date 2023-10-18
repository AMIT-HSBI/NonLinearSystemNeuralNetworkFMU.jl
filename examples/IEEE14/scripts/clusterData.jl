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
import Clustering
import Distances


function readData(filename::String, nInputs::Integer; ratio=0.8, shuffle::Bool=true)
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
    return train_in, train_out, test_in, test_out
end


fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/data/eq_1403.csv"
nInputs = 16
nOutputs = 110

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

function vectorofvector_to_matrix(vov)
    return mapreduce(permutedims, vcat, vov)
end

train_in = vectorofvector_to_matrix(train_in)
train_out = vectorofvector_to_matrix(train_out)

train_in_t = transpose(train_in)
train_out_t = transpose(train_out)


dt = StatsBase.fit(StatsBase.ZScoreTransform, train_out_t, dims=1) # normalise along columns
train_out_t = StatsBase.transform(dt, train_out_t)


max_score = 0
max_score_num_cluster = 1
max_cluster = 20
distances = Distances.pairwise(Distances.SqEuclidean(), train_out_t)
for i = 2:max_cluster
    R = Clustering.kmeans(train_out_t, i; maxiter=200)
    score = mean(Clustering.silhouettes(R, distances))
    if score > max_score
        max_score = score
        max_score_num_cluster = i
    end
end
print(max_score)
print(max_score_num_cluster)


R = Clustering.kmeans(train_out_t, 2; maxiter=200)

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

# Example usage:
#cluster_assignments = [1, 2, 1, 3, 4, 2, 1, 3, 4]
result = get_cluster_indices(R.assignments)
println(result)


function extract_clusters(data::Matrix{Float32}, cluster_indices::Vector{Vector{Int64}}, cluster_index::Int64)
    # data is in the form (d, n)
    # d - feature dimension
    # n - number of datapoints
    # cluster_indices: index of clusters
    # cluster_index: index of cluster to get
    return data[:,cluster_indices[cluster_index]]
end

