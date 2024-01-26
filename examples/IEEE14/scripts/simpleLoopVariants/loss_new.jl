#TODO: maybe dispatch on batchsize

function loss(y_hat, fmu, eq_num, sys_num, transform)
    bs = size(y_hat)[2] # batchsize
    residuals = Array{Vector{Float64}}(undef, bs)
    for j in 1:bs
        yj_hat = StatsBase.reconstruct(transform, y_hat[:,j])
        _, res = fmiEvaluateRes(fmu, eq_num, Float64.(yj_hat))
        residuals[j] = res
    end
    return 1/(2*bs)*sum(norm.(residuals).^2), if bs>1 residuals else residuals[1] end
end

function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform)
    l, res = loss(x, fmu, eq_num, sys_num, transform)
    # evaluate the jacobian for each batch element
    bs = size(x)[2] # batchsize
    res_dim = if bs>1 length(res[1]) else length(res) end
    jac_dim = trunc(Int,sqrt(res_dim))

    jacobians = Array{Matrix{Float64}}(undef, bs)
    for j in 1:bs
        xj = x[:,j]
        _, jac = fmiEvaluateJacobian(comp, sys_num, vr, Float64.(xj))
        jacobians[j] = reshape(jac, (jac_dim,jac_dim))
    end

    function loss_pullback(l̄)
        l_tangent = l̄[1] # upstream gradient
        factor = l_tangent/bs

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
