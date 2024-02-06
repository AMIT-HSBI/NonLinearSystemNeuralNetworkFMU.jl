#TODO: maybe dispatch on batchsize

function loss(y_hat, fmu, eq_num, sys_num, transform)
    bs = size(y_hat)[2] # batchsize
    residuals = Array{Vector{Float64}}(undef, bs)
    for j in 1:bs
        _, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat[:,j]))
        residuals[j] = res
    end
    return 1/(2*bs)*sum(norm.(residuals).^2), if bs>1 residuals else residuals[1] end
end

function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform)
    l, res = loss(x, fmu, eq_num, sys_num, transform)
    # evaluate the jacobian for each batch element
    bs = size(x)[2] # batchsize
    res_dim = if bs>1 length(res[1]) else length(res) end
    jac_dim = res_dim

    jacobians = Array{Matrix{Float64}}(undef, bs)
    for j in 1:bs
        _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(x[:,j]))
        jacobians[j] = reshape(jac, (jac_dim,jac_dim))
    end

    function loss_pullback(l̄)
        l_tangent = l̄[1] # upstream gradient
        factor = l_tangent/bs # factor should probably be just: factor=l_tangent!!!!

        x̄ = Array{Float64}(undef, res_dim, bs)
        # compute x̄
        for j in 1:bs
            x̄[:,j] = transpose(jacobians[j]) * res[j]
        end
        x̄ *= factor
    
        # all other args have NoTangent
        f̄ = NoTangent()
        fmū = NoTangent()
        eq_num̄ = NoTangent()
        sys_num̄ = NoTangent()
        transform̄ = NoTangent()
        return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
    end

    return l, loss_pullback
end



function loss(y_hat, fmu, eq_num, sys_num, transform)
    # transform is either train_out_transform or test_out_transform
    """
    y_hat is model output
    evaluates residual of system eq_num at y_hat
    if y_hat is close to a solution, residual is close to 0
    actual loss is the norm of the residual
    """
    batchsize = size(y_hat)[2]
    if batchsize>1
      residuals = []
      for i in 1:batchsize
        y_hat_i = y_hat[1:end,i]
        y_hat_i_rec = StatsBase.reconstruct(transform, y_hat_i)
        _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat_i_rec))
        push!(residuals, res_out)
      end
      return mean((1/2).*LinearAlgebra.norm.(residuals).^2), residuals
    else
      y_hat_rec = StatsBase.reconstruct(transform, y_hat)
      _, res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat_rec))
      return 1/2*(LinearAlgebra.norm(res_out)^2), res_out
    end
    # p = 0
    # if y_hat_rec[1] < 0
    #   p = abs(y_hat_rec[1])
    # end
  end
  
  
  # rrule for loss(x,y)
  #BATCH
  function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform)
    """
    reverse rule for loss function
    needs the jacobian of the system eq_num evaluated at x (x is model output)
    uses that formula: https://math.stackexchange.com/questions/291318/derivative-of-the-2-norm-of-a-multivariate-function
    """
    l, res_out = loss(x, fmu, eq_num, sys_num, transform) # res_out: residual output, what shape is that?
    # evaluate the jacobian for each batch element
    batchsize = size(x)[2]
    if batchsize>1
      jacobians = []
      for i in 1:batchsize
        x_i = x[1:end,i]
        _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(x_i))
        mat_dim = trunc(Int,sqrt(length(jac)))
        jac = reshape(jac, (mat_dim,mat_dim))
        push!(jacobians, jac)
      end
    else
      _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(x[:,1]))
      
      mat_dim = trunc(Int,sqrt(length(jac)))
      jac = reshape(jac, (mat_dim,mat_dim))
    end
  
    function loss_pullback(l̄)
      l_tangent = l̄[1] # upstream gradient
  
      # compute x̄
      if batchsize>1
        # backprop through mean of norms of batch elements
        factor = l_tangent/batchsize
        x̄ = jacobians[1]' * res_out[1]
        for i in 2:batchsize
          x̄ = x̄ + jacobians[i]' * res_out[i]
        end
        x̄*=factor
        x̄ = repeat(x̄, 1, batchsize)
      else
        x̄ = l_tangent * (jac' * res_out)
      end
  
      # all other args have NoTangent
      f̄ = NoTangent()
      fmū = NoTangent()
      eq_num̄ = NoTangent()
      sys_num̄ = NoTangent()
      transform̄ = NoTangent()
      return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
    end
    return l, loss_pullback
  end