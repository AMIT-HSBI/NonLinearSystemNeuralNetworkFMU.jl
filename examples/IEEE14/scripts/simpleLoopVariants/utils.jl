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
  
    vr = FMI.fmiStringToValueReference(fmu, profilinginfo[1].iterationVariables)
  
    eq_num = profilinginfo[1].eqInfo.id
    sys_num = parse_modelfile(model_path, eq_num)
  
  
    row_value_reference = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].usingVars)
  
    return comp, fmu, profilinginfo, vr, row_value_reference, sys_num
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


# plot x and y
function plot_xy(model, in_data, out_data)
    scatter(compute_x_from_y.(in_data[1,:],in_data[2,:],vec(out_data)), vec(out_data))
    prediction = model(in_data)
    scatter!(compute_x_from_y.(in_data[1,:],in_data[2,:],vec(prediction)), vec(prediction))
end

function plot_loss_history(loss_history; kwargs...)
    x = 1:length(loss_history)
    plot(x, loss_history; kwargs...)
  end