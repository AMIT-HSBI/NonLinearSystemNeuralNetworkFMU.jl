using Surrogates


num_samples = 10
lb = [0.0, 0.0]
ub = [5.0, 5.0]

function generateDataPoint(fmu, eqId, nInputs, row_vr, row, time)
    # Set input values and start values for output
    FMIImport.fmi2SetReal(fmu, row_vr, row)
    if time !== nothing
      FMIImport.fmi2SetTime(fmu, time)
    end
  
    # Evaluate equation
    # TODO: Supress stream prints, but Suppressor.jl is not thread safe
    status = fmiEvaluateEq(fmu, eqId)
    if status == fmi2OK
      # Get output values
      row[nInputs+1:end] .= FMIImport.fmi2GetReal(fmu, row_vr[nInputs+1:end])
    end
  
    return status, row
end

#Sampling
x = sample(num_samples,lb,ub,SobolSample())
function g(fmu, eqId, nInputs, nOutputs, row_vr, inputs, time)
    _, in_out = generateDataPoint(fmu, eqId, nInputs, row_vr, vcat(inputs, zeros(nOutputs)), time)
    return in_out[nInputs+1:end]
end
# actual function to be optimized
f = x -> g(fmu, eqId, nInputs, nOutputs, row_vr, x, time)
y = f.(x)

#Creating surrogate
methods = ["RadialBasis", "SecondOrderPolynomialSurrogate"]

surrogate = RadialBasis()

value = surrogate([0.0, 0.0])

#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,surrogate,RandomSample())

#New approximation
value = surrogate([0.0, 0.0])