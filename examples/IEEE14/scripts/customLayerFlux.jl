# implementation of prelu activation function as a layer in Flux
# alpha is a learnable parameter


using Flux


struct prelu{Float64}
    alpha::Float64
    function prelu(alpha_init::Float64 = 0.25)
        new{Float64}(alpha_init)
    end
end
  
function (a::prelu)(x::AbstractArray)
    pos = Flux.relu(x)
    neg = -a.alpha * Flux.relu(-x)
    return pos + neg
end

Flux.@functor prelu

pr = prelu()


pr([1,2,3]) # [1.0, 2.0, 3.0]
pr([1,-2,3]) # [1.0, -0.5, 3.0]