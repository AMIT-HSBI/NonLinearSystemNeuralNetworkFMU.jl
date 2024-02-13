# function loss(y_hat, fmu, eq_num, sys_num, transform)
#   bs = size(y_hat)[2] # batchsize
#   residuals = Array{Vector{Float64}}(undef, bs)
#   for j in 1:bs
#       yj_hat = StatsBase.reconstruct(transform, y_hat[:,j])
#       _, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(yj_hat))
#       residuals[j] = res
#   end
#   return 1/(2*bs)*sum(norm.(residuals).^2), if bs>1 residuals else residuals[1] end
# end

# function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform)
#   l, res = loss(x, fmu, eq_num, sys_num, transform)
#   # evaluate the jacobian for each batch element
#   bs = size(x)[2] # batchsize
#   res_dim = if bs>1 length(res[1]) else length(res) end
#   jac_dim = res_dim

#   jacobians = Array{Matrix{Float64}}(undef, bs)
#   for j in 1:bs
#       xj = StatsBase.reconstruct(transform, x[:,j])
#       _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(xj))
#       jacobians[j] = reshape(jac, (jac_dim,jac_dim))
#   end

#   function loss_pullback(l̄)
#       l_tangent = l̄[1] # upstream gradient
#       factor = l_tangent/bs # factor should probably be just: factor=l_tangent!!!!

#       x̄ = Array{Float64}(undef, res_dim, bs)
#       # compute x̄
#       for j in 1:bs
#           x̄[:,j] = transpose(jacobians[j]) * res[j]
#       end
#       x̄ = if transform.dims == 1 x̄ .* (1 ./ transform.scale)' elseif transform.dims == 2 x̄ .* (1 ./ transform.scale) end
#       x̄ *= factor
  
#       # all other args have NoTangent
#       f̄ = NoTangent()
#       fmū = NoTangent()
#       eq_num̄ = NoTangent()
#       sys_num̄ = NoTangent()
#       transform̄ = NoTangent()
#       return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
#   end

#   return l, loss_pullback
# end


function trainModelSemisupervised(model, optimizer, train_dataloader, test_in, test_out, train_in_transform, test_in_transform, train_out_transform,  test_out_transform, eq_num, sys_num, row_value_reference, fmu; epochs=100, h1=1.0, h2=0.2)
  #https://physicsbaseddeeplearning.org/physicalloss.html
    opt_state = Flux.setup(optimizer, model)
    test_loss_history = []

    t0 = time()
    for epoch in 1:epochs
      for (x,y) in train_dataloader
          prepare_x(x, row_value_reference, fmu, train_in_transform)
          lv, grads = Flux.withgradient(model) do m  
            prediction = m(x)
            h1 * Flux.mse(prediction, y) + h2 * (loss(prediction, fmu, eq_num, sys_num, train_out_transform))
          end
          Flux.update!(opt_state, model, grads[1])
      end
      push!(test_loss_history, Flux.mse(model(test_in), test_out))
    end
    t1 = time()
    training_time = t1 - t0

    return model, test_loss_history, training_time
end


