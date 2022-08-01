#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

function omrun(cmd::Cmd; dir=pwd()::String)

  path = ";" * abspath(joinpath(ENV["OPENMODELICAHOME"], "tools", "msys", "mingw64", "bin"))
  path *= ";" * abspath(joinpath(ENV["OPENMODELICAHOME"], "tools", "msys", "usr", "bin"))

  run(Cmd(cmd, env=("PATH" => path,), dir = dir))
end

"""
Create C files for extendedFMU with special_interface to call single equations
"""
function createSpecialInterface(modelname::String, tempDir::String, eqIndices::Array{Int64})
  # Open template
  path = joinpath(@__DIR__,"templates", "special_interface.tpl.h")
  hFileContent = open(path) do file
    read(file, String)
  end

  # Replace placeholders
  hFileContent = replace(hFileContent, "<<MODELNAME>>"=>modelname)

  # Create `special_interface.h`
  path = joinpath(tempDir,"FMU", "sources", "fmi-export", "special_interface.h")
  open(path, "w") do file
    write(file, hFileContent)
  end

  # Open template
  path = joinpath(@__DIR__,"templates", "special_interface.tpl.c")
  cFileContent = open(path) do file
    read(file, String)
  end

  # Replace placeholders
  cFileContent = replace(cFileContent, "<<MODELNAME>>"=>modelname)
  forwardEquationBlock = ""
  for eqIndex in eqIndices
    forwardEquationBlock = forwardEquationBlock *
      """extern void $(modelname)_eqFunction_$(eqIndex)(DATA* data, threadData_t *threadData);"""
  end
  cFileContent = replace(cFileContent, "<<FORWARD_EQUATION_BLOCK>>"=>forwardEquationBlock)
  equationCases = ""
  for eqIndex in eqIndices
    equationCases = equationCases *
      """
        case $(eqIndex):
          $(modelname)_eqFunction_$(eqIndex)(data, threadData);
          break;
      """
  end
  cFileContent = replace(cFileContent, "<<EQUATION_CASES>>"=>equationCases)

  # Create `special_interface.c`
  path = joinpath(tempDir,"FMU", "sources", "fmi-export", "special_interface.c")
  open(path, "w") do file
    write(file, cFileContent)
  end
end

"""
Generate 2.0 Model Exchange FMU for Modelica model using omc.
"""
function generateFMU(;modelName::String, pathToMo::String, pathToOmc::String, tempDir::String, clean::Bool = false)

  if !isdir(tempDir)
    mkdir(tempDir)
  elseif clean
    rm(tempDir, force=true, recursive=true)
    mkdir(tempDir)
  end

  if Sys.iswindows()
    pathToMo = replace(pathToMo, "\\"=> "\\\\")
    tempDir = replace(tempDir, "\\"=> "\\\\")
  end

  logFilePath = joinpath(tempDir,"callsFMI.log")
  logFile = open(logFilePath, "w")

  omc = OMJulia.OMCSession(pathToOmc)
  try
    msg = OMJulia.sendExpression(omc, "getVersion()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "loadFile(\"$(pathToMo)\")")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "cd(\"$(tempDir)\")")

    @info "setCommandLineOptions"
    msg = OMJulia.sendExpression(omc, "setCommandLineOptions(\"-d=newInst\")")
    write(logFile, string(msg)*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")

    @info "buildFMU"
    msg = OMJulia.sendExpression(omc, "buildModelFMU($(modelName), version=\"2.0\", fmuType=\"me\")")
    write(logFile, msg*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
  finally
    close(logFile)
    OMJulia.sendExpression(omc, "quit()",parsed=false)
  end

  if !isfile(joinpath(tempDir, modelName*".fmu"))
    error("Could not generate FMU! Check log file:\n$(abspath(logFilePath))")
  end

  return joinpath(tempDir, modelName*".fmu")
end


function updateMakefile(path_to_makefile::String; clean::Bool = false, debug::Bool = true)
  newStr = ""
  open(path_to_makefile, "r") do file
    filestr = read(file, String)

    # Update CPPFLAGS
    id1 = first(findfirst("CPPFLAGS=", filestr))
    id1 = first(findnext(' ', filestr, id1))-1
    newStr = filestr[1:id1]*" -Ifmi-export"*filestr[id1+1:end]

    # Add fmi-export/special_interface.c to CFILES
    id2 = first(findfirst("OFILES=\$(CFILES:.c=.o)", newStr)) - 2
    # TODO: Add whitespace
    newStr = newStr[1:id2]*" \\\n"*"  fmi-export/special_interface.c"*newStr[id2+1:end]

    if !clean
      # Deactivate distclean rule
      id2 = last(findfirst("distclean: clean", newStr)) + 1
      newStr = newStr[1:id2]*"#"*newStr[id2+1:end]

      # Deactivate clean rule
      id2 = last(findnext("clean:", newStr, id2)) + 1
      newStr = newStr[1:id2]*"#"*newStr[id2+1:end]
    end

    if debug
      newStr = replace(newStr, "-O2"=>"-O0")
    end
  end

  write(path_to_makefile, newStr)
end

# Function inspired by FMIImport.fmi2Unzip
# See: https://github.com/ThummeTo/FMIImport.jl/blob/2eb951018e0d2530349a17e6040f1f82ecfb220f/src/FMI2_ext.jl#L13-L79
"""
    unzip(zipFile, targetDir)

# Arguments
  - `zipFile::String`: Path to zip file.
  - `targetDir::String`: Directory to extract into.

"""
function unzip(zipFile::String, targetDir::String)
  # Create target directory
  if !isdir(targetDir)
    mkpath(targetDir)
  end

  # Unzip all files
  archive = ZipFile.Reader(abspath(zipFile))
  for file in archive.files
    fileAbsPath = normpath(joinpath(targetDir, file.name))

    if endswith(file.name,"/") || endswith(file.name,"\\")
      mkpath(fileAbsPath)
    else
      mkpath(dirname(fileAbsPath))
      write(fileAbsPath, read(file))
      @assert(isfile(fileAbsPath), "Couldn't create $(file)")
    end
  end
  close(archive)
end


"""
Create extendedFMU with special_interface
"""
function addEqInterface2FMU(;modelName::String,
                             pathToFmu::String,
                             pathToFmiHeader::String,
                             eqIndices::Array{Int64},
                             tempDir::String)

  @info "Unzip FMU"
  pathToFmu = abspath(pathToFmu)
  @assert(isfile(pathToFmu), "FMU $(pathToFmu) not found.")
  fmuPath = abspath(joinpath(tempDir,"FMU"))
  unzip(pathToFmu, fmuPath)

  # make special_interface in FMU/sources/fmi-export
  @info "Add special C sources"
  modelname = replace(modelName, "."=>"_")
  createSpecialInterface(modelname, abspath(tempDir), eqIndices)

  # Configure in FMU/sources while adding FMI headers to `CPPFLAGS`
  @info "Configure"
  if !Sys.iswindows()
    run(Cmd(`./configure CPPFLAGS="-I$(pathToFmiHeader)" NEED_CMINPACK=1`, dir = "$(fmuPath)/sources"))
  end

  # Update Makefile
  if Sys.iswindows()
    updateMakefile(joinpath(fmuPath,"sources/Makefile"), clean=true)
  else
    updateMakefile(joinpath(fmuPath,"sources/Makefile"))
  end

  # run make
  @info "Compiling FMU"
  if Sys.iswindows()
    omrun(`make -sj $(modelname)_FMU`, dir = "$(fmuPath)/sources")
  else
    run(Cmd(`make -sj $(modelname)_FMU`, dir = "$(fmuPath)/sources"))
  end

  return joinpath(tempDir, modelName*".fmu")
end
