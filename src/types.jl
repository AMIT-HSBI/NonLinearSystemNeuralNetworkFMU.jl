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

 #=
 #   General types
=#

"""
    EqInfo <: Any

Equation info struct.

$(DocStringExtensions.TYPEDFIELDS)
"""
struct EqInfo
  "Unique equation id"
  id::Int64
  "Number of calls during simulation"
  ncall::Int64
  "Total time [s] spend on evaluating this equation."
  time::Float64
  "Maximum time [s] needed for single evaluation of equation."
  maxTime::Float64
  "Fraction of total simulation time spend on evaluating this equation."
  fraction::Float64

  # Constructors
  EqInfo(id, ncall, time, maxTime, fraction) = new(id, ncall, time, maxTime, fraction)
  EqInfo(;id, ncall, time, maxTime, fraction) = new(id, ncall, time, maxTime, fraction)
end

function Base.show(io::IO, ::MIME"text/plain", eqInfo::EqInfo)
  Printf.@printf(io, "EqInfo(id=%s, ", eqInfo.id)
  Printf.@printf(io, "ncall=%s, ", eqInfo.ncall)
  Printf.@printf(io, "time=%s, ", eqInfo.time)
  Printf.@printf(io, "maxTime=%s, ", eqInfo.maxTime)
  Printf.@printf(io, "fraction=%s)", eqInfo.fraction)
end

"""
    MinMaxBoundaryValues <: Any

Minimum and maximum boundary values of list of variables.
"""
struct MinMaxBoundaryValues{T}
  "Minimum boundary values."
  min::Array{T, 1}
  "Maximum boundary values."
  max::Array{T, 1}

  # Constructors
  function MinMaxBoundaryValues{T}(init::UndefInitializer, N) where T
    min = Array{T}(init, N)
    max = Array{T}(init, N)
    new{T}(min, max)
  end
  function MinMaxBoundaryValues(min::Array{T,1}, max::Array{T,1}) where T
    if size(min) != size(max)
      throw(ArgumentError("Wrong size: min and max argument need to have same size."))
    end
    for i in 1:length(min)
      if min[i] > max[i]
        throw(ArgumentError("Wrong order: $i-th minimum element bigger than maximum."))
      end
    end
    new{T}(min, max)
  end
end

"""
    ProfilingInfo <: Any

Profiling information for single non-linear equation.

$(DocStringExtensions.TYPEDFIELDS)
"""
struct ProfilingInfo
  "Non-linear equation"
  eqInfo::EqInfo
  "Iteration (output) variables of non-linear system"
  iterationVariables::Array{String}
  "Inner (torn) equations of non-linear system."
  innerEquations::Array{Int64}
  "Used (input) variables of non-linear system."
  usingVars::Array{String}
  "Minimum and maximum boundary values of `usingVars`."
  boundary::MinMaxBoundaryValues{Float64}
end

function Base.show(io::IO, ::MIME"text/plain", profilingInfo::ProfilingInfo)
  Printf.@printf(io, "ProfilingInfo(%s, ", profilingInfo.eqInfo)
  Printf.@printf(io, "iterationVariables: %s, ", profilingInfo.iterationVariables)
  Printf.@printf(io, "innerEquations: %s, ", profilingInfo.innerEquations)
  Printf.@printf(io, "usingVars: %s, ", profilingInfo.usingVars)
  Printf.@printf(io, "boundary: %s)", profilingInfo.boundary)
end


"""
    OMOptions <: Any

Settings for profiling and simulating with the OpenModelica Compiler (OMC).
"""
struct OMOptions
  "Path to omc used for simulating the model."
  pathToOmc::String
  "Working directory for omc. Defaults to the current directory."
  workingDir::String
  """Output format for result file. Can be `"mat"` or `"csv"`."""
  outputFormat::Union{String,Nothing}
  "Remove everything in `workingDir` when set to `true`."
  clean::Bool
  "Additional comannd line options for `setCommandLineOptions`."
  commandLineOptions::String

  # Constructor
  function OMOptions(;pathToOmc::String = "",
                     workingDir::String = pwd(),
                     outputFormat::Union{String,Nothing} = "csv",
                     clean::Bool = false,
                     commandLineOptions::String = "",
                     disableCSE = true)

    # Try to find omc executable
    pathToOmc = getomc(pathToOmc)

    # Assert output format
    if outputFormat != "csv" && outputFormat != "mat" && outputFormat !== nothing
      error("output format $(outputFormat) not supperted. Has to be \"csv\" or \"mat\".")
    end

    # Disable CSE variables in loops
    if disableCSE
      commandLineOptions *= " --preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls"
    end

    new(pathToOmc, workingDir, outputFormat, clean, commandLineOptions)
  end
end

 #=
 #   Error types
=#

"""
Program not found in PATH error.
"""
struct ProgramNotFoundError <: Exception
  program::String
  locations::Union{Nothing, Array{String}}
  ProgramNotFoundError(program) = new(program, nothing)
end
function Base.showerror(io::IO, e::ProgramNotFoundError)
  println(io, e.program, " not found")
  if e.locations !== nothing
    for loc in e.locations
      println(io, "Searched in " * loc)
    end
  end
end

"""
Minimum version not sattisfied error.
"""
struct MinimumVersionError <: Exception
  program::String
  minimumVersion::String
  currentVersion::String
end
function Base.showerror(io::IO, e::MinimumVersionError)
  println(io, e.program, " minimum version requierment not satisfied.")
  println(io, "Minimum version needed is $(e.minimumVersion).")
  print(io, "Using version $(e.currentVersion).")
end

"""
String not found in searched string error.
"""
struct StringNotFoundError <: Exception
  searchString::String
end
function Base.showerror(io::IO, e::StringNotFoundError)
  print(io, "Could not find string \"", e.searchString, " \"")
end

"""
OpenModelica error with log file.
"""
struct OpenModelicaError <: Exception
  msg::String
  logFile::String
end
function Base.showerror(io::IO, e::OpenModelicaError)
  println(io, e.msg)
  println(io, "Log file: ", e.logFile)
  if e.logFile !== nothing &&isfile(e.logFile)
    println(io, "Printing log file: ", e.logFile)
    print(io, read(e.logFile, String))
  else
    println(io, "No log file")
  end
end

"""
Timeout error.
"""
struct TimeOutError <: Exception
  cmd::Cmd
end
function Base.showerror(io::IO, e::TimeOutError)
  println(io, "Timeout reached running command")
  println(io, e.cmd)
end

 #=
 #   Getter and setter functions
=#

function getProfilingInfo(bsonFile::String)::Array{ProfilingInfo}
  # load BSON file
  dict = BSON.load(bsonFile, @__MODULE__)
  return Array{ProfilingInfo}(dict[first(keys(dict))])
end

"""
    getUsingVars(bsonFile, eqNumber)

# Arguments
  - `bsonFile::String`:  name of the binary JSON file
  - `eqNumber::Int`:  number of the slowest equation
# Return:
  - array of the using variables and length of the array
"""
function getUsingVars(bsonFile::String, eqNumber::Int)
  profilingInfo = getProfilingInfo(bsonFile)
  # Find array element with eqNumber
  for prof in profilingInfo
    if prof.eqInfo.id == eqNumber
      return prof.usingVars , length(prof.usingVars)
    end
  end
end

"""
    getIterationVars(bsonFile, eqNumber)

# Arguments
  - `bsonFile::String`:  name of the binary JSON file
  - `eqNumber::Int`:  number of the slowest equation
# Return:
  - array of the iteration variables and length of the array
"""
function getIterationVars(bsonFile::String, eqNumber::Int)
  profilingInfo = getProfilingInfo(bsonFile)
  # Find array element with eqNumber
  for prof in profilingInfo
    if prof.eqInfo.id == eqNumber
      return prof.iterationVariables , length(prof.iterationVariables)
    end
  end
end

"""
    getInnerEquations(bsonFile, eqNumber)

# Arguments
  - `bsonFile::String`:  name of the binary JSON file
  - `eqNumber::Int`:  number of the slowest equation
# Return:
  - array of the inner equations and length of the array
"""
function getInnerEquations(bsonFile::String, eqNumber::Int)
  profilingInfo = getProfilingInfo(bsonFile)
  # Find array element with eqNumber
  for prof in profilingInfo
    if prof.eqInfo.id == eqNumber
      return prof.innerEquations, length(prof.innerEquations)
    end
  end
end

"""
    getMinMax(bsonFile, eqNumber, inputArray)

# Arguments
  - `bsonFile::String`:  name of the binary JSON file
  - `eqNumber::Int`:  number of the slowest equation
  - `inputArray::Vector{String}`: array of input variables as String
# Return:
  - array of the min and max values of each input from input array
"""
function getMinMax(bsonFile::String, eqNumber::Int, inputArray::Vector{String})
  profilingInfo = getProfilingInfo(bsonFile)
  for prof in profilingInfo
    if prof.eqInfo.id == eqNumber
      # get using variables; length is not necessary
      inputs = prof.usingVars
      # compare strings of inputArray with strings of inputs
      indices = indexin(inputs,inputArray)
      if length(inputs) > length(inputArray)
        deleteat!(indices, findall(x -> x === nothing,indices))
      end
      return [[min,max] for (min,max) in zip(prof.boundary.min[indices],prof.boundary.max[indices])]
    end
  end
end

"""
    getMinMax(bsonFile, eqNumber, inputArray)

# Arguments
  - `bsonFile::String`:  name of the binary JSON file
  - `eqNumber::Int`:  number of the slowest equation
  - `inputArray::Vector{Int}`: array of input variables as Integers
# Return:
  - array of the min and max values of each input from input array
"""
function getMinMax(bsonFile::String, eqNumber::Int, inputArray::Vector{Int})
  profilingInfo = getProfilingInfo(bsonFile)
  for prof in profilingInfo
    if prof.eqInfo.id == eqNumber
      return [[min,max] for (min,max) in zip(prof.boundary.min[inputArray],prof.boundary.max[inputArray])]
    end
  end
end
