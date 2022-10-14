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

"""
    testCMakeVersion(;minimumVersion = "3.21")

Test if CMake can be found in PATH and minimum version is satisfied.
Returns used version or throw an error if minimum version is not satisfied.
"""
function testCMakeVersion(;minimumVersion = "3.21")
  # Find cmake
  try
    if Sys.iswindows()
      run(pipeline(`where cmake.exe`; stdout=devnull))
    else
      run(pipeline(`which cmake`; stdout=devnull))
    end
  catch
    throw(ProgramNotFoundError("cmake", [ENV["PATH"]]))
  end

  # Get version string
  local version
  out = IOBuffer()
  err = IOBuffer()
  try
    run(pipeline(`cmake --version`; stdout=out, stderr=err))
    version = String(take!(out))
    version = split(strip(version))[1:3]
    @assert version[1] == "cmake"
    @assert version[2] == "version"
    version = version[3]
  catch e
    println(String(take!(err)))
    rethrow(e)
  finally
    close(out)
    close(err)
  end

  if version < minimumVersion
    throw(MinimumVersionError("CMake", minimumVersion, version))
  end

  return version
end

"""
    testOmcVersion(omc;  minimumVersion = "v1.20.0-dev-342")

Test if OMC version is higher than minimum version.
Throw error if omc can't be found or minimum version is not satisfied.
"""
function testOmcVersion(omc; minimumVersion = "v1.20.0-dev-342")
  # Check if path points to a file
  if !isfile(omc)
    throw(ProgramNotFoundError("omc", [ENV["PATH"]]))
  end
  @debug "Using omc: $omc"

  # Get version string
  local version
  out = IOBuffer()
  err = IOBuffer()
  try
    run(pipeline(`omc --version`; stdout=out, stderr=err))
    version = strip(String(take!(out)))
  catch e
    println(String(take!(err)))
    rethrow(e)
  finally
    close(out)
    close(err)
  end

  def_min = "9999"
  main_ver_min = minimumVersion
  loc = findfirst("dev", minimumVersion)
  if loc !== nothing
    (main_ver_min, def_min) = split(minimumVersion, "-")[1:2:3]
  end
  if startswith(main_ver_min, "v")
    main_ver_min = main_ver_min[2:end]
  end

  def_omc = "0"
  main_ver_omc = version
  loc = findfirst("dev", version)
  if loc !== nothing
    (main_ver_omc, def_omc) = split(version, "-")[1:2:3]
  end
  if startswith(main_ver_omc, "v")
    main_ver_omc = main_ver_omc[2:end]
  end

  if main_ver_omc < main_ver_min
    @show main_ver_omc, main_ver_min
    throw(MinimumVersionError("omc", minimumVersion, version))
  elseif main_ver_omc == main_ver_min && def_omc < def_min
    @show def_min, def_omc
    throw(MinimumVersionError("omc", minimumVersion, version))
  end
end

