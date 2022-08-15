#
# Copyright (c) 2022 Andreas Heuermann, Philip Hannebohm
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

struct Mapping
  name::String
  valueReference::String
  #type::String
end

EOL = Sys.iswindows() ? "\r\n" : "\n"

function ortDataCode(equations::Array{ProfilingInfo}, modelName::String)
  ortstructs = ""
  initCalls = ""
  deinitCalls = ""
  nEq = length(equations)
  for (i,eq) in enumerate(equations)
    onnxName = "$(modelName)_eq$(eq.eqInfo.id).onnx"
    nInputs = length(eq.usingVars)
    nOutputs = length(eq.iterationVariables)
    # TODO: Get input and output names
    input_name = "data_0"
    output_name = "dense_2"
    ortstructs *= "struct OrtWrapperData* ortData_eq_$(eq.eqInfo.id);"
    initCalls *= """
        snprintf(onnxPath, 2048, "%s/%s", data->modelData->resourcesDir, \"$onnxName\");
        ortData_eq_$(eq.eqInfo.id) = initOrtData(onnxPath, \"$modelName\", $nInputs, $nOutputs, \"$input_name\", \"$output_name\");
      """
    deinitCalls *= "  deinitOrtData(ortData_eq_$(eq.eqInfo.id));"
    if i < nEq
      ortstructs *= "$EOL  "
      deinitCalls *= "$EOL  "
    end
    if i == nEq
      initCalls = initCalls[1:end-1]
    end
  end

  code = """
    #include "onnxWrapper.h"

    int USE_JULIA = 1;

    /* Global ORT structs */
    $(ortstructs)

    /* Init function */
    void initGlobalOrtData(DATA* data) {
      char onnxPath[2048];
    $(initCalls)
    }

    /* Deinit function */
    void deinitGlobalOrtData() {
    $(deinitCalls)
    }
    """
  return code
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

function generateNNCall(modelname::String, modelDescriptionXmlFile::String, equationToReplace::ProfilingInfo)
  variablesDict = getValueReferences(modelDescriptionXmlFile)

  inputs = equationToReplace.usingVars
  outputs = equationToReplace.iterationVariables

  inputVarBlock = ""
  for (i,var) in enumerate(inputs)
    cVar = getVarCString(var, variablesDict)
    inputVarBlock *= "input[$(i-1)] = $(cVar);"
    if i < length(inputs)
      inputVarBlock *= "$EOL    "
    end
  end

  outputVarBlock = ""
  for (i,var) in enumerate(outputs)
    cVar = getVarCString(var, variablesDict)
    outputVarBlock *= "$(cVar) = output[$(i-1)];"
    if i < length(outputs)
      inputVarBlock *= "$EOL    "
    end
  end

  innerEquations = ""
  for (i,eq) in enumerate(equationToReplace.innerEquations)
    innerEquations *= "$(modelname)_eqFunction_$(eq)(data, threadData);"
    if i < length(equationToReplace.innerEquations)
      innerEquations *= "$EOL    "
    end
  end

  ortData = "ortData_eq_$(equationToReplace.eqInfo.id)"

  cCode = """
      float* input = $ortData->model_input;
      float* output = $ortData->model_output;

      $inputVarBlock

      evalModel($ortData);

      $outputVarBlock

      /* Eval inner equations */
      $innerEquations
  """

  return cCode
end

function modifyCCode(modelName::String, cfile::String, modelDescriptionXmlFile::String, equations::Array{ProfilingInfo})

  str = open(cfile, "r") do file
    read(file, String)
  end

  # Add init/ deinint ortData
  id1 = first(findfirst("/* dummy VARINFO and FILEINFO */", str)) - 2
  initCode = ortDataCode(equations, modelName)
  str = str[1:id1] * initCode * str[id1+1:end]

  id1 = last(findfirst("$(modelName)_setupDataStruc(DATA *data, threadData_t *threadData)", str))
  id1 = last(findnext("data->modelData->nExtObjs", str, id1))
  id1 = last(findnext(";$EOL", str, id1))
  str = str[1:id1] * "$EOL  initGlobalOrtData(data);$EOL" * str[id1+1:end]

  # Replace nls-call in equation
  for equation in equations
    eqInfo = equation.eqInfo
    id1 = last(findfirst("$(modelName)_eqFunction_$(eqInfo.id)(DATA *data, threadData_t *threadData)", str))
    id1 = first(findnext("/* get old value */", str, id1)) - 1
    id2 = first(findnext("  TRACE_POP", str, id1)) -1

    oldpart = str[id1:id2]
    oldpart = replace(oldpart, "$EOL  "=>"$EOL    ")
    newpart = generateNNCall(modelName, modelDescriptionXmlFile, equation)

    replacement = """
    if(USE_JULIA) {
    $newpart
      } else {
        $oldpart
      }
    """
    str = str[1:id1] * replacement * str[id2:end]
  end

  write(cfile, str)
end


function modifyMakefile(makefile::String, ortdir::String, fmuBinaryDir::String)

  str = open(makefile, "r") do file
    read(file, String)
  end

  # Set include flag
  includedir = "-I\"$(ortdir)/include/\" "
  str = replace(str, "CPPFLAGS="=>"CPPFLAGS=$(includedir)")

  str = replace(str, ".interface.fmu"=>".onnx.fmu")

  # Set linker flags
  extraLdflags = """
    LDFLAGS += -L\"$(ortdir)/lib/\" -lonnxruntime \\
               -L\"$(fmuBinaryDir)\" -lonnxWrapper '-Wl,-rpath,\$\$ORIGIN'

    """
  id1 = first(findfirst("PHONY:", str)) - 1
  str = str[1:id1] * extraLdflags * str[id1+1:end]
  write(makefile, str)
end

"""
    getFmuBinDir()

Get FMU binary directory.

For example nn 64bit Windwos this is `"binaries/win64"`.
On 32bit Linux this is `"binaries/linux32`.
"""
function getFmuBinDir()
  if Sys.iswindows()
    return joinpath("binaries", "linux"*string(Sys.WORD_SIZE))
  elseif Sys.islinux()
    return joinpath("binaries", "win"*string(Sys.WORD_SIZE))
  elseif Sys.isapple()
    return joinpath("binaries", "darwin"*string(Sys.WORD_SIZE))
  else
    error("OS not handled.")
  end
end

function copyOnnxWrapperLib(fmuRootDir::String)
  # Copy header file
  hfilesource = joinpath(@__DIR__, "onnxWrapper", "install", "include", "onnxWrapper.h")
  @assert isfile(hfilesource)
  hfiledest = joinpath(fmuRootDir, "sources", "onnxWrapper.h")
  cp(hfilesource, hfiledest)

  # Copy library
  libdir = joinpath(@__DIR__, "onnxWrapper", "install", "lib")
  @assert isdir(libdir)
  for source in readdir(libdir)
    libdest = joinpath(fmuRootDir, getFmuBinDir(), basename(source))
    cp(joinpath(libdir, source), libdest)
  end
end

function copyOnnxFiles(fmuRootDir::String, onnxFiles::Array{String})
  resourcesDir = joinpath(fmuRootDir, "resources")
  @assert isdir(resourcesDir)
  for file in onnxFiles
    cp(file, joinpath(resourcesDir, basename(file)))
  end
end

function buildWithOnnx(fmu::String, modelName::String, equations::Array{ProfilingInfo}, onnxFiles::Array{String}, ortdir::String; tempDir=modelName*"_onnx"::String)

  # Unzip FMU into tmp dir
  fmuTmpDir = abspath(joinpath(tempDir,"FMU"))
  rm(fmuTmpDir, force=true, recursive=true)
  unzip(fmu, fmuTmpDir)

  cfile = joinpath(fmuTmpDir, "sources", "$(modelName).c")
  modelDescriptionXmlFile = joinpath(fmuTmpDir, "modelDescription.xml")
  makefile = joinpath(fmuTmpDir, "sources", "Makefile")

  modifyMakefile(makefile, ortdir, getFmuBinDir())
  copyOnnxWrapperLib(fmuTmpDir)
  copyOnnxFiles(fmuTmpDir, onnxFiles)
  modifyCCode(modelName, cfile, modelDescriptionXmlFile, equations)
  compileFMU(fmuTmpDir, modelName)

  return joinpath(tempDir, "$(modelName).onnx.fmu")
end
