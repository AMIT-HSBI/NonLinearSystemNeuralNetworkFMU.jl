#
# Copyright (c) 2022 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
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
    generateFMU(;modelName::String,
                 pathToMo::String,
                 pathToOmc::String, 
                 tempDir::String,
                 clean::Bool = false)

Generate 2.0 Model Exchange FMU for Modelica model using OMJulia.

# Keywords
  - `modelName::String`: Name of Modelica model to export as FMU.
  - `pathToMo::String`: Path to Modelica file.
  - `pathToOmc::String`: Path to OpenModlica Compiler.
  - `tempDir::String`: Path to temp directory in which FMU will be saved to.
  - `clean::Bool=false`: True if tempDir should be removed and re-created before working in it.

# Returns
  - Path to generated FMU `tempDir/<modelName>.fmu`.

See also [`addEqInterface2FMU`](@ref), [`generateTrainingData`](@ref).
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
    id1 = last(findfirst("CPPFLAGS=", filestr))
    newStr = filestr[1:id1] * " -Ifmi-export " * filestr[id1+1:end]

    # Add fmi-export/special_interface.c to CFILES
    id2 = first(findfirst("OFILES=\$(CFILES:.c=.o)", newStr)) - 1
    if Sys.iswindows()
      newStr = newStr[1:id2-2]*" \\\r\n         fmi-export/special_interface.c\r\n"*newStr[id2+1:end]
    else
      newStr = newStr[1:id2-1]*" \\\n         fmi-export/special_interface.c\n"*newStr[id2+1:end]
    end

    newStr = replace(newStr, ".fmu"=>".interface.fmu")

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

"""
    unzip(file, exdir)

Unzip `file` to directory `exdir`.
"""
function unzip(file::String, exdir::String)
  @assert(isfile(file), "File $(file) not found.")
  if !isdir(exdir)
    mkpath(exdir)
  end

  if Sys.iswindows()
    omrun(`unzip -q -o $(file) -d $(exdir)`)
  else
    run(Cmd(`unzip -q -o $(file) -d $(exdir)`))
  end
end


"""
    compileFMU(fmuRootDir, modelname)

Run `fmuRootDir/sources/Makefile` to compile FMU binaries.
"""
function compileFMU(fmuRootDir::String, modelname::String)
  # run make
  @info "Compiling FMU"
  if Sys.iswindows()
    omrun(`make -sj $(modelname)_FMU`, dir = joinpath(fmuRootDir,"sources"))
  else
    run(Cmd(`make -sj $(modelname)_FMU`, dir = joinpath(fmuRootDir,"sources")))
  end
end

"""
    addEqInterface2FMU(;modelName::String,
                        pathToFmu::String,
                        pathToFmiHeader::String,
                        eqIndices::Array{Int64},
                        tempDir::String)

Create extendedFMU with special_interface to evalaute single equations.

# Keywords
  - `modelName::String`: Name of Modelica model to export as FMU.
  - `pathToFmu::String`: Path to FMU to extend.
  - `pathToFmiHeader::String`: Path to FMI headers.
                               They are part of this repository in FMI-Standard-2.0.3/headers.
  - `eqIndices::Array{Int64}`: Array with equation indices to add equiation interface for.
  - `tempDir::String`:

# Returns
  - Path to generated FMU `tempDir/<modelName>.interface.fmu`.

See also [`profiling`](@ref), [`generateFMU`](@ref), [`generateTrainingData`](@ref).
"""
function addEqInterface2FMU(;modelName::String,
                             pathToFmu::String,
                             pathToFmiHeader::String,
                             eqIndices::Array{Int64},
                             tempDir::String)

  @info "Unzip FMU"
  fmuPath = abspath(joinpath(tempDir,"FMU"))
  unzip(abspath(pathToFmu), fmuPath)

  # make special_interface in FMU/sources/fmi-export
  @info "Add special C sources"
  modelname = replace(modelName, "."=>"_")
  createSpecialInterface(modelname, abspath(tempDir), eqIndices)

  pathToFmiHeader = abspath(pathToFmiHeader)

  # Configure in FMU/sources while adding FMI headers to `CPPFLAGS`
  @info "Configure"
  if !Sys.iswindows()
    run(Cmd(`./configure CPPFLAGS="-I$(pathToFmiHeader)" NEED_CMINPACK=1`, dir = joinpath(fmuPath,"sources")))
  end

  # Update Makefile
  if Sys.iswindows()
    updateMakefile(joinpath(fmuPath,"sources", "Makefile"), clean=true)
  else
    updateMakefile(joinpath(fmuPath,"sources", "Makefile"))
  end

  compileFMU(fmuPath, modelName)

  return joinpath(tempDir, "$(modelName).interface.fmu")
end
