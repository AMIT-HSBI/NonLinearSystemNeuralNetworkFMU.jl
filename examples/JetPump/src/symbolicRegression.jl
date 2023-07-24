using SymbolicRegression
using SymbolicUtils
using SymbolicUtils.Code

function fitEquation(X, Y; tempDir)
  # Change to temp directory
  previousDir = pwd()
  if !isdir(tempDir)
    mkpath(tempDir)
  end

  options = SymbolicRegression.Options(
    binary_operators = [+, *, /, -],
    unary_operators  = [sqrt],
    npopulations     = 20
  )

  # Run sybolic regression
  local hall_of_fame
  try
    cd(tempDir)

    hall_of_fame = EquationSearch(
      X, Y,
      niterations = 100,
      options     = options,
      parallelism = :multithreading
    )
  finally
    cd(previousDir)
  end

  # Convert best equation to SymbolicUtils
  bestEq = Any[]
  for (i,_) in enumerate(eachrow(Y))
    dominating = calculate_pareto_frontier(hall_of_fame[i])
    bestEqDim = node_to_symbolic(dominating[end].tree, options)
    bestEqDim = simplify(bestEqDim)
    push!(bestEq, bestEqDim)
  end

  return bestEq
end
