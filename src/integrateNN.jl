#
# Copyright (c) 2022 Andreas Heuermann, Philip Hannebohm
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#


#=
1. Mapping between model inputs / output variables and OpenModelica FMU variables
2. Generate calling code for ONNX
3. Wrapp NN in ONNX
4. Add ONNX and wrapper code to FMU

=#

struct Mapping
  name::String
  valueReference::String
  #type::String
end

function getValueReferences(modelDescriptionXML::String)::Dict{String, Mapping}
  xml = open(modelDescriptionXML) do file
    XMLDict.parse_xml(read(file, String))
  end

  dict = Dict{String, Mapping}()

  variables = xml["ModelVariables"]["ScalarVariable"]

  for var in variables
    dict[var[:name]] = Mapping(var[:name], var[:valueReference])
  end

  return dict
end

function getVarCString(varName::String, variables::Dict{String, Mapping})
  str = "data->localData[0]->realVars[$(variables[varName].valueReference)] /* $(varName) */"

  return str
end


function generateNNCall(modelDescriptionXmlFile::String, equationToReplace::ProfilingInfo)
  variablesDict = getValueReferences(modelDescriptionXmlFile)

  inputs = equationToReplace.usingVars
  outputs = equationToReplace.iterationVariables

  inputVarBlock = ""
  for (i,var) in enumerate(inputs)
    cVar = getVarCString(var, variablesDict)
    inputVarBlock *= "input[$(i-1)] = $(cVar);"
    if i < length(inputs)
      inputVarBlock *= "\n  "
    end
  end

  outputVarBlock = ""
  for (i,var) in enumerate(outputs)
    cVar = getVarCString(var, variablesDict)
    outputVarBlock *= "$(cVar) = output[$(i-1)];"
    if i < length(outputs)
      inputVarBlock *= "\n  "
    end
  end

  cCode = """
    float* input = inputDataPtr(ortData);
    float* output = outputDataPtr(ortData);

    $inputVarBlock

    evalModel(ortData);

    $outputVarBlock
  """

  return cCode
end

function addNNCall(modelName::String, cfile::String, modelDescriptionXmlFile::String, equationToReplace::ProfilingInfo)

  str = open(cfile, "r") do file
    read(file, String)
  end

  eq = equationToReplace.eqInfo
  @show "void $(modelName)_eqFunction_$(eq.id)(DATA *data, threadData_t *threadData)"


  id1 = last(findfirst("$(modelName)_eqFunction_$(eq.id)(DATA *data, threadData_t *threadData)", str))
  id1 = first(findnext("/* get old value */", str, id1)) - 1

  id2 = first(findnext("  TRACE_POP", str, id1)) -1

  oldpart = str[id1:id2]
  newpart = generateNNCall(modelDescriptionXmlFile, equationToReplace)

  replacement = """
    if(USE_JULIA) {
      $newpart
    } else {
      $oldpart
    }
  """

  str = str[1:id1] * replacement * str[id2:end]

  #print(str)
  write(cfile, str)
end