function loss(y_hat, fmu, eq_num, sys_num, transform)
  bs = size(y_hat)[2] # batchsize
  residuals = Array{Vector{Float64}}(undef, bs)
  for j in 1:bs
      _, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat[:,j]))
      residuals[j] = res
  end
  return 1/(2*bs)*sum(norm.(residuals).^2), if bs>1 residuals else residuals[1] end
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