import Zygote
import StatsBase
import ChainRulesCore

x = rand(4,3)
function f(x, transf)
    return StatsBase.reconstruct(transf, x)
end

dt = StatsBase.fit(StatsBase.UnitRangeTransform, x, dims=2)

xx = StatsBase.transform(dt, x)
f(xx, dt)


# rrule for reconstruct
function ChainRulesCore.rrule(::typeof(f), x, transf)
    y = f(x, transf)
    function f_pullback(ȳ)
        f̄ = ChainRulesCore.NoTangent()
        x̄ = if transf.dims == 1 ȳ .* (1 ./ transf.scale)' elseif transf.dims == 2 ȳ .* (1 ./ transf.scale) end
        transf̄ = ChainRulesCore.NoTangent()
        return f̄, x̄, transf̄
    end
    return y, f_pullback
end


y, back = ChainRulesCore.rrule(f, x, dt)

_, x_bar, _ = back(ones(size(x)))
x_bar
x


dx = zeros(size(x))
dx[4,3] = 1e-6
(f(xx + dx, dt) - f(xx, dt)) / 1e-6
