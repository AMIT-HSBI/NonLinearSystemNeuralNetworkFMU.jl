#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

"""
Create C files for extendedFMU with special_interface to call single equations
"""
function createSpecialInterface(modelname::String, tempDir::String, eqIndices::Array{Int64})
  # create `special_interface.h`
  sihFilePath = joinpath(tempDir,"FMU", "sources", "fmi-export", "special_interface.h")
  sihFile = open(sihFilePath, "w")
  write(sihFile,"""
    #include \"../simulation_data.h\"
    #include \"../simulation/solver/solver_main.h\"
    #include \"../$(modelname)_model.h\"
    #include \"fmu2_model_interface.h\"
    #include \"fmu_read_flags.h\"

    fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber);
    """)
  close(sihFile)

  # create `special_interface.c`
  sicFilePath = joinpath(tempDir,"FMU/sources/fmi-export/special_interface.c")
  sicFile = open(sicFilePath, "w")
  write(sicFile,
    """
    #include \"../simulation_data.h\"
    #include \"../simulation/solver/solver_main.h\"
    #include \"../$(modelname)_model.h\"
    #include \"fmu2_model_interface.h\"
    #include \"fmu_read_flags.h\"

    fmi2Boolean isCategoryLogged(ModelInstance *comp, int categoryIndex);

    static fmi2String logCategoriesNames[] = {\"logEvents\", \"logSingularLinearSystems\", \"logNonlinearSystems\", \"logDynamicStateSelection\", \"logStatusWarning\", \"logStatusDiscard\", \"logStatusError\", \"logStatusFatal\", \"logStatusPending\", \"logAll\", \"logFmi2Call\"};

    #ifndef FILTERED_LOG
      #define FILTERED_LOG(instance, status, categoryIndex, message, ...) if (isCategoryLogged(instance, categoryIndex)) { \\
          instance->functions->logger(instance->functions->componentEnvironment, instance->instanceName, status, \\
              logCategoriesNames[categoryIndex], message, ##__VA_ARGS__); }
    #endif

    /* forwarded equations */
    """)
  for eqIndex in eqIndices
    write(sicFile,
      "extern void $(modelname)_eqFunction_$(eqIndex)(DATA* data, threadData_t *threadData);\n")
  end
  write(sicFile,
    """
    fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber)
    {
      ModelInstance *comp = (ModelInstance *)c;
      DATA* data = comp->fmuData;
      threadData_t *threadData = comp->threadData;

      useStream[LOG_NLS] = 0 /* false */;
      useStream[LOG_NLS_V] = 0 /* false */;
      FILTERED_LOG(comp, fmi2OK, LOG_FMI2_CALL, \"myfmi2evaluateEq: Evaluating equation %u\", eqNumber)

      switch (eqNumber)
      {
    """)
  for eqIndex in eqIndices
    write(sicFile,
    """
        case $(eqIndex):
          $(modelname)_eqFunction_$(eqIndex)(data, threadData);
          comp->_need_update = 0;
          break;
    """)
  end
  write(sicFile,
    """
      default:
        return fmi2Error;
        break;
      }

      return fmi2OK;
    }""")
  close(sicFile)
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


function updateMakefile(path_to_makefile::String)
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

    # Deactivate distclean rule
    id2 = last(findfirst("distclean: clean", newStr)) + 1
    newStr = newStr[1:id2]*"#"*newStr[id2+1:end]
  end

  write(path_to_makefile, newStr)
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
  fmuPath = abspath(joinpath(tempDir,"FMU"))
  run(`bash -c "unzip -q -o $(pathToFmu) -d $(fmuPath)"`)

  # make special_interface in FMU/sources/fmi-export
  @info "Add special C sources"
  modelname = replace(modelName, "."=>"_")
  createSpecialInterface(modelname, abspath(tempDir), eqIndices)

  # Configure in FMU/sources while adding FMI headers to `CPPFLAGS`
  @info "Configure"
  run(`bash -c "cd $(fmuPath)/sources ; ./configure CPPFLAGS="-I$(pathToFmiHeader)" NEED_CMINPACK=1"`)

  # update Makefile
  updateMakefile(joinpath(fmuPath,"sources/Makefile"))

  # run make
  @info "Compiling FMU"
  run(`bash -c "cd $(fmuPath)/sources ; make -sj -C $(fmuPath)/sources/ $(modelname)_FMU"`)

  return joinpath(tempDir, modelName*".fmu")
end
