using NonLinearSystemNeuralNetworkFMU
import Flux
import Zygote
import StatsBase
import ChainRulesCore
using LinearAlgebra

# x = rand(4,3)
# function f(x, transf)
#     return StatsBase.reconstruct(transf, x)
# end

# dt = StatsBase.fit(StatsBase.UnitRangeTransform, x, dims=2)

# xx = StatsBase.transform(dt, x)
# f(xx, dt)


# # rrule for reconstruct
# function ChainRulesCore.rrule(::typeof(f), x, transf)
#     y = f(x, transf)
#     function f_pullback(ȳ)
#         f̄ = ChainRulesCore.NoTangent()
#         x̄ = if transf.dims == 1 ȳ .* (1 ./ transf.scale)' elseif transf.dims == 2 ȳ .* (1 ./ transf.scale) end
#         transf̄ = ChainRulesCore.NoTangent()
#         return f̄, x̄, transf̄
#     end
#     return y, f_pullback
# end


# y, back = ChainRulesCore.rrule(f, x, dt)

# _, x_bar, _ = back(ones(size(x)))
# x_bar
# x


# dx = zeros(size(x))
# dx[1,1] = 1e-6
# (f(xx + dx, dt) - f(xx, dt)) / 1e-6


function loss(y_hat, fmu, eq_num, sys_num, transform)
    bs = size(y_hat)[2] # batchsize
    residuals = Array{Vector{Float64}}(undef, bs)
    for j in 1:bs
        yj_hat = StatsBase.reconstruct(transform, y_hat[:,j])
        _, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(yj_hat))
        residuals[j] = res
    end
    return 1/(2*bs)*sum(norm.(residuals).^2), residuals
end


include("utils.jl")
# data loading into 
fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/data/eq_14.csv"
nInputs = 2
nOutputs = 1
comp, fmu, profilinginfo, vr, row_value_reference, eq_num, sys_num = prepare_fmu("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson",
                                                            "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/temp-profiling/simpleLoop.c")
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs) 
# min max scaling
train_in, train_out, test_in, test_out, train_in_t, train_out_t, test_in_t, test_out_t = scale_data_uniform(train_in, train_out, test_in, test_out)
dataloader = Flux.DataLoader((train_in, train_out), batchsize=8, shuffle=true)


x, y = first(dataloader)

function ChainRulesCore.rrule(::typeof(loss), x, fmu, eq_num, sys_num, transform, finitediff)
    l, res = loss(x, fmu, eq_num, sys_num, transform)
    # evaluate the jacobian for each batch element
    bs = size(x)[2] # batchsize
    res_dim = length(res[1])

    if finitediff == false
        jac_dim = res_dim

        jacobians = Array{Matrix{Float64}}(undef, bs)
        for j in 1:bs
            xj = StatsBase.reconstruct(transform, x[:,j])
            _, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, sys_num, vr, Float64.(xj))
            jacobians[j] = reshape(jac, (jac_dim,jac_dim))
        end

        function loss_pullback_analytical(l̄)
            println("analytical")
            factor = l̄./bs

            x̄ = Array{Float64}(undef, res_dim, bs)
            # compute x̄
            for j in 1:bs
                x̄[:,j] = transpose(jacobians[j]) * res[j]
            end
            x̄ = if transform.dims == 1 x̄ .* (1 ./ transform.scale)' elseif transform.dims == 2 x̄ .* (1 ./ transform.scale) end
            x̄ .*= factor

            # all other args have NoTangent
            f̄ = ChainRulesCore.NoTangent()
            fmū = ChainRulesCore.NoTangent()
            eq_num̄ = ChainRulesCore.NoTangent()
            sys_num̄ = ChainRulesCore.NoTangent()
            transform̄ = ChainRulesCore.NoTangent()
            return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
        end
        return l, loss_pullback_analytical
    else
        function loss_pullback_finitediff(l̄)
            println("finitediff")
            factor = l̄

            x̄ = Array{Float64}(undef, res_dim, bs)
        
            for i in 1:res_dim
                for j in 1:bs
                    dx = zeros((res_dim, bs));
                    dx[i,j] = 1e-5;
                    x̄[i,j] = (loss(x + dx, fmu, eq_num, sys_num, train_out_t)[1] - loss(x, fmu, eq_num, sys_num, train_out_t)[1]) / 1e-5
                end
            end

            x̄ .*= factor

            # all other args have NoTangent
            f̄ = ChainRulesCore.NoTangent()
            fmū = ChainRulesCore.NoTangent()
            eq_num̄ = ChainRulesCore.NoTangent()
            sys_num̄ = ChainRulesCore.NoTangent()
            transform̄ = ChainRulesCore.NoTangent()
            return (f̄, x̄, fmū, eq_num̄, sys_num̄, transform̄)
        end
        return l, loss_pullback_finitediff
    end
end

l, res = loss(y, fmu, eq_num, sys_num, train_out_t)

l, back_analytical = ChainRulesCore.rrule(loss, y, fmu, eq_num, sys_num, train_out_t)

grad_analytical = back_analytical(ones(size(y)))[2]

l, res = loss(y, fmu, eq_num, sys_num, train_out_t)

l, back_finitediff = ChainRulesCore.rrule(loss, y, fmu, eq_num, sys_num, train_out_t)

grad_finitediff = back_analytical(ones(size(y)))[2]


















finitediff_der = zeros(size(y));
for i in 1:size(y)[1]
    for j in 1:size(y)[2]
        dy = zeros(size(y));
        dy[i,j] = 1e-4;
        finitediff_der[i,j] = (loss(y + dy, fmu, eq_num, sys_num, train_out_t)[1] - loss(y, fmu, eq_num, sys_num, train_out_t)[1]) / 1e-4
    end
end
finitediff_der


# finite diff and rrule match for batchsize=1 but not for other batchsizes

NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y[:,1]))

loss(y, fmu, eq_num, sys_num, train_out_t)
