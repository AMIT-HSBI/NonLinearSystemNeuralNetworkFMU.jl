using NonLinearSystemNeuralNetworkFMU
using JSON


function replaceSurrogates(flatModel::String, prof::ProfilingInfo, infoJsonFile::String, eqIndex::Int, surrogatEqns::Array{String})

  # Load info file
  infoFile = JSON.parsefile(infoJsonFile)
  equations = infoFile["equations"]
  nlsequation = (equations[eqIndex+1])

  if nlsequation["eqIndex"] != eqIndex
    error("Found wrong equation")
  end

  # Get residual equations
  residualEqns = []
  for i in nlsequation["equation"][1]
    eq = equations[i+1]
    if eq["tag"] == "residual"
      push!(residualEqns, eq)
    end
  end

  # Replace residual equations
  sort!(residualEqns, by = x -> x["source"]["info"]["lineStart"])

  content = readlines(flatModel)

  for (i, resEq) in enumerate(residualEqns)
    lineStart = resEq["source"]["info"]["lineStart"]
    lineEnd = resEq["source"]["info"]["lineEnd"]
    colStart = resEq["source"]["info"]["colStart"]
    colEnd = resEq["source"]["info"]["colEnd"]
    @assert lineStart == lineEnd "Line start and end are not equal"

    line = content[lineStart]
    newline = line[1:colStart-1] * surrogatEqns[i] * line[colEnd+1:end]   # TODO: It seems thas surrogatEqns has to be in the correct order...
    content[lineStart] = newline
  end

  surrFile = splitext(flatModel)[1]*"_sur.mo"
  open(surrFile,"w") do file
    write(file, join(content, "\n"))
  end

  return surrFile
end
