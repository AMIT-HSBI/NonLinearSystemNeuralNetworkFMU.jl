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


profinfo_path = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson"
profilinginfo = getProfilingInfo(profinfo_path)


times = []
for i in [1000, 10000, 20000]
    t1 = time()
    dm = generate_unsupervised_data(profilinginfo, i)
    println(size(dm))
    t2 = time()
    push!(times, t2 - t1)
end

times