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
  str = "data->localData[0]->realVars[$(variables.valueReference)] /* $(varName) */"

  return str
end


function generateNNCall(modelDescriptionXmlFile::String, equationToReplace::ProfilingInfo)

  variablesDict = getValueReferences(modelDescriptionXmlFile)

  inputs = equationToReplace.usingVars
  outputs = equationToReplace.iterationVariables

  inputVarBlock = ""
  for (i,var) in enumerat(inputs)
    cVar = getVarCString(var, variablesDict)
    inputVarBlock *= """
                     input[$i] = $(cVar);
                     """
  end

  outputVarBlock = ""
  for (i,var) in enumerat(outputs)
    cVar = getVarCString(var, variablesDict)
    outputVarBlock *= """
                      $(cVar) = output[$i];
                      """
  end

  cCode = """
  float* input = inputDataPtr(ortData);
  float* output = outputDataPtr(ortData);

  $inputVarBlock

  evalModel(ortData);

  $outputVarBlock
  """
end

#infoJsonFile = abspath(joinpath(@__DIR__, "..", "/test/simpleLoop/simpleLoop_info.json"))
modelDescriptionXML = abspath("test/fmus/FMU/modelDescription.xml")