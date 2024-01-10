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

function trainModelSupervised(model, optimizer, train_dataloader, test_in, test_out, train_in_transform, test_in_transform, train_out_transform,  test_out_transform, eq_num, sys_num, row_value_reference, fmu; epochs=100)
    ps = Flux.params(model)
    opt_state = Flux.setup(optimizer, model)
    test_loss_history = []
    res_test_loss_history = []
    training_time = 0
    
    for epoch in 1:epochs
        t0 = time()
        for (x, y) in train_dataloader
            lv, grads = Flux.withgradient(model) do m  
                prediction = m(x)
                Flux.mse(prediction, y)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        t1 = time()
        training_time += t1 - t0

        push!(test_loss_history, Flux.mse(model(test_in), test_out))
        prepare_x(test_in, row_value_reference, fmu, test_in_transform)
        l,_ = loss(model(test_in), fmu, eq_num, sys_num, test_out_transform)
        push!(res_test_loss_history, l)
    end
    return model, test_loss_history, res_test_loss_history, training_time
end