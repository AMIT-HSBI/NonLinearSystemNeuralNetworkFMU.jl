#
# Copyright (c) 2022-2023 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
#

EOL = Sys.iswindows() ? "\r\n" : "\n"

"""
    omrun(cmd; dir=pwd(), logFile=devnull, timeout=10*60)

Execute system command.
Kill process and throw TimeOutError when timeout is reached.
Catch InterruptException, kill process and rethorw InterruptException.

Add OPENMODELICAHOME to PATH for Windows to get access to Unix tools from MSYS.

# Arguments
  - `cmd::Cmd`:             Shell command to run.
  - `dir=pwd()::String`:    Working directory for command.
  - `logFile=stdout`:       IO stream or file to pipe stdout and stderr to.
                            Will append if file already exists.

# Keywords
  - `timeout=10*60::Integer`:   Timeout in seconds. Defaults to 10 minutes.
"""
function omrun(cmd::Cmd; dir=pwd()::String, logFile=stdout, timeout=10*60::Integer)
  env = copy(ENV)
  if Sys.iswindows()
    path = env["PATH"]
    if !haskey(ENV, "OPENMODELICAHOME")
      error("Environment variable `OPENMODELICAHOME` not set.")
    end
    local omdev_msys
    if haskey(ENV, "OMDEV_MSYS")
      omdev_msys = ENV["OMDEV_MSYS"]
    else
      omdev_msys = joinpath(ENV["OPENMODELICAHOME"], "tools", "msys")
    end
    if !isdir(omdev_msys)
      error("Couldn't find MSYS environment from OpenModelica. " *
            "Check if environment variable `OPENMODELICAHOME` is pointing to the correct location.")
    end
    if isdir(joinpath(omdev_msys, "ucrt64", "bin"))
      path *= ";" * abspath(joinpath(omdev_msys, "ucrt64", "bin"))
      path *= ";" * abspath(joinpath(omdev_msys, "usr", "bin"))
    elseif isdir(joinpath(omdev_msys, "mingw64", "bin"))
      path *= ";" * abspath(joinpath(omdev_msys, "mingw64", "bin"))
      path *= ";" * abspath(joinpath(omdev_msys, "usr", "bin"))
    end
    env["PATH"] = path
    env["CLICOLOR"] = "0"
  end
  @debug "PATH: $(env["PATH"])"

  cmd_path = Cmd(cmd, env=env, dir = dir)
  append = true
  if logFile == stdout || logFile == stderr || logFile == devnull
    append = false
  end
  plp = pipeline(cmd_path, stdout=logFile, stderr=logFile, append=append)
  p = run(plp, wait=false)
  try
    timer = Timer(0; interval=1)
    for i in 1:timeout
      wait(timer)
      if process_running(p)
        @debug "Command still running: $i[s]"
      else
        @debug "Finished command"
        close(timer)
        break
      end
    end
    if process_running(p)
      @debug "Killing $(p)"
      kill(p)
      throw(TimeOutError(cmd))
    end
  catch e
    if isa(e, InterruptException) && process_running(p)
      @error "Killing process $(cmd)."
      kill(p)
    end
    rethrow(e)
  end
  if p.exitcode != 0
    error("Process $(cmd) failed")
  end
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
  residualCases = ""
  for eqIndex in eqIndices
    residualCases = residualCases *
      """
        case $(eqIndex):
          residualFunc$(eqIndex)(&resUserData, x, res, &iflag);
          break;
      """
  end
  cFileContent = replace(cFileContent, "<<RESIDUAL_CASES>>"=>residualCases)

  # Create `special_interface.c`
  path = joinpath(tempDir,"FMU", "sources", "fmi-export", "special_interface.c")
  open(path, "w") do file
    write(file, cFileContent)
  end
end


"""
    generateFMU(modelName, moFiles; options)

Generate 2.0 Model Exchange FMU for Modelica model using OMJulia.

# Arguments
  - `modelName::String`:        Name of the Modelica model.
  - `moFiles::Array{String}`:   Path to the *.mo file(s) containing the model.

# Keywords
  - `options::OMOptions`:       Options for OpenModelica compiler.

# Returns
  - Path to generated FMU `workingDir/<modelName>.fmu`.

See also [`OMOptions`](@ref), [`addEqInterface2FMU`](@ref), [`generateTrainingData`](@ref).
"""
function generateFMU(modelName::String,
                     moFiles::Array{String};
                     options::OMOptions)

  # Create / clean working diretory
  workingDir = options.workingDir
  if !isdir(workingDir)
    mkpath(workingDir)
  elseif options.clean
    rm(workingDir, force=true, recursive=true)
    mkpath(workingDir)
  end

  logFilePath = joinpath(workingDir,"callsFMI.log")
  logFile = open(logFilePath, "w")

  local omc
  Suppressor.@suppress begin
    omc = OMJulia.OMCSession(options.pathToOmc)
  end
  try
    version = OMJulia.API.getVersion(omc)
    write(logFile, version*"\n")
    for file in moFiles
      OMJulia.API.loadFile(omc, file)
    end
    OMJulia.API.cd(omc, workingDir)

    @debug "setCommandLineOptions"
    OMJulia.API.setCommandLineOptions(omc, "-d=newInst --fmiFilter=internal --fmuCMakeBuild=true --fmuRuntimeDepends=modelica " * options.commandLineOptions)

    @debug "buildFMU"
    OMJulia.API.buildModelFMU(omc, modelName; version="2.0", fmuType="me", platforms=["dynamic"])
  catch e
    @error "Failed to build FMU for $modelName."
    rethrow(e)
  finally
    close(logFile)
    OMJulia.quit(omc)
  end

  if !isfile(joinpath(workingDir, modelName*".fmu"))
    throw(OpenModelicaError("Could not generate FMU!", abspath(logFilePath)))
  end

  return joinpath(workingDir, modelName*".fmu")
end


function updateCMakeLists(path_to_cmakelists::String)
  newStr = ""
  open(path_to_cmakelists, "r") do file
    filestr = read(file, String)
    id1 = last(findStrWError("\${CMAKE_CURRENT_SOURCE_DIR}/external_solvers/*.c", filestr))
    newStr = filestr[1:id1] * EOL *
             "                              \${CMAKE_CURRENT_SOURCE_DIR}/fmi-export/*.c" *
             filestr[id1+1:end]
  end

  write(path_to_cmakelists, newStr)
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

  omrun(`unzip -q -o $(file) -d $(exdir)`)
end


"""
    compileFMU(fmuRootDir, modelname, workdir)

Run `fmuRootDir/sources/CMakeLists.txt` to compile FMU binaries.
Needs CMake version 3.21 or newer.
"""
function compileFMU(fmuRootDir::String, modelname::String, workdir::String)
  testCMakeVersion()

  @debug "Compiling FMU"
  logFile = joinpath(workdir, modelname*"_compile.log")
  rm(logFile, force=true)
  @info "Compilation log file: $(logFile)"

  if !haskey(ENV, "ORT_DIR")
    @warn "Environment variable ORT_DIR not set."
  elseif !isdir(ENV["ORT_DIR"])
    @warn "Environment variable ORT_DIR not pointing to a directory."
    @show ENV["ORT_DIR"]
  end

  try
    pathToFmiHeader = abspath(joinpath(dirname(@__DIR__), "FMI-Standard-2.0.3", "headers"))
    if Sys.iswindows()
      omrun(`cmake -S . -B build_cmake -Wno-dev -G "MSYS Makefiles" -DCMAKE_COLOR_MAKEFILE=OFF`, dir = joinpath(fmuRootDir,"sources"), logFile=logFile)
      omrun(`mingw32-make install -Oline -j`, dir = joinpath(fmuRootDir, "sources", "build_cmake"), logFile=logFile)
    else
      omrun(`cmake -S . -B build_cmake`, dir = joinpath(fmuRootDir,"sources"), logFile=logFile)
      omrun(`cmake --build build_cmake/ --target install --parallel`, dir = joinpath(fmuRootDir, "sources"), logFile=logFile)
    end
    rm(joinpath(fmuRootDir, "sources", "build_cmake"), force=true, recursive=true)
    # Use create_zip instead of calling zip
    rm(joinpath(dirname(fmuRootDir),modelname*".fmu"), force=true)
    omrun(`zip -r ../$(modelname).fmu binaries/ resources/ sources/ modelDescription.xml`, dir = fmuRootDir, logFile=logFile)
  catch e
    @info "Error caught, dumping log file $(logFile)"
    println(read(logFile, String))
    rethrow(e)
  end
end


"""
    addEqInterface2FMU(modelName, pathToFmu, eqIndices; workingDir=pwd())

Create extendedFMU with special_interface to evalaute single equations.

# Arguments
  - `modelName::String`:        Name of Modelica model to export as FMU.
  - `pathToFmu::String`:        Path to FMU to extend.
  - `eqIndices::Array{Int64}`:  Array with equation indices to add equiation interface for.

# Keywords
  - `workingDir::String=pwd()`: Working directory. Defaults to current working directory.

# Returns
  - Path to generated FMU `workingDir/<modelName>.interface.fmu`.

See also [`profiling`](@ref), [`generateFMU`](@ref), [`generateTrainingData`](@ref).
"""
function addEqInterface2FMU(modelName::String,
                            pathToFmu::String,
                            eqIndices::Array{Int64};
                            workingDir::String=pwd())

  @debug "Unzip FMU"
  fmuPath = abspath(joinpath(workingDir,"FMU"))
  unzip(abspath(pathToFmu), fmuPath)

  # make special_interface in FMU/sources/fmi-export
  @debug "Add special C sources"
  modelname = replace(modelName, "."=>"_")
  createSpecialInterface(modelname, abspath(workingDir), eqIndices)

  # Update CMakeLists.txt
  updateCMakeLists(joinpath(fmuPath,"sources", "CMakeLists.txt"))

  # Re-compile FMU
  compileFMU(fmuPath, modelName*".interface", workingDir)

  return joinpath(workingDir, "$(modelName).interface.fmu")
end
