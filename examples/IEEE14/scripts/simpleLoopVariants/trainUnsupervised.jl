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


function trainModelUnsupervised(model, optimizer, train_dataloader, test_in, test_out, train_in_transform, test_in_transform, train_out_transform,  test_out_transform, eq_num, sys_num, row_value_reference, fmu; epochs=100)
    # if out data is nothing i dont record supervised loss
    # else i do
    ps = Flux.params(model)
    opt_state = Flux.setup(optimizer, model)
    test_loss_history = []
    res_test_loss_history = []
    training_time = 0

    for epoch in 1:epochs
      # training
      t0 = time()
      for (x,y) in train_dataloader
          prepare_x(x, row_value_reference, fmu, train_in_transform)
          lv, grads = Flux.withgradient(model) do m  
            prediction = m(x)
            loss(prediction, fmu, eq_num, sys_num, train_out_transform)
          end
          Flux.update!(opt_state, model, grads[1])
      end
      t1 = time()
      training_time += t1 - t0

      # test loss
      push!(test_loss_history, Flux.mse(model(test_in), test_out))
      prepare_x(test_in, row_value_reference, fmu, test_in_transform)
      l,_ = loss(model(test_in), fmu, eq_num, sys_num, test_out_transform)
      push!(res_test_loss_history, l)
    end

    return model, test_loss_history, res_test_loss_history, training_time
end