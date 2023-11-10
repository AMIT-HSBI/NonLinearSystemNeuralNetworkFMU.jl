var documenterSearchIndex = {"docs":
[{"location":"dataGen/#Training-Data-Generation","page":"Data Generation","title":"Training Data Generation","text":"","category":"section"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"To generate training data for the slowest non-linear equations found during Profiling Modelica Models we now simulate the equations multiple time and save in- and outputs.","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"We will use the Functional Mock-up Interface (FMI) standard to generate FMU that we extend with some function to evaluate single equations without the need to simulate the rest of the model.","category":"page"},{"location":"dataGen/#Functions","page":"Data Generation","title":"Functions","text":"","category":"section"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"generateTrainingData\naddEqInterface2FMU\ngenerateFMU","category":"page"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.generateTrainingData","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.generateTrainingData","text":"generateTrainingData(fmuPath, workDir, fname, eqId, inputVars, min, max, outputVars;\n                     options=DataGenOptions())\n\nGenerate training data for given equation of FMU.\n\nGenerate random inputs between min and max, evalaute equation and compute output. All input-output pairs are saved in CSV file fname.\n\nArguments\n\nfmuPath::String:                Path to FMU.\nworkDir::String:                Working directory for generateTrainingData.\nfname::String:                  File name to save training data to.\neqId::Int64:                    Index of equation to generate training data for.\ninputVars::Array{String}:       Array with names of input variables.\nminBound::AbstractVector{T}:    Array with minimum value for each input variable.\nmaxBound::AbstractVector{T}:    Array with maximum value for each input variable.\noutputVars::Array{String}:      Array with names of output variables.\n\nKeywords\n\noptions::DataGenOptions:        Settings for data generation.\n\nSee also generateFMU, DataGenOptions..\n\n\n\n\n\n","category":"function"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.addEqInterface2FMU","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.addEqInterface2FMU","text":"addEqInterface2FMU(modelName, pathToFmu, eqIndices; workingDir=pwd())\n\nCreate extendedFMU with special_interface to evalaute single equations.\n\nArguments\n\nmodelName::String:        Name of Modelica model to export as FMU.\npathToFmu::String:        Path to FMU to extend.\neqIndices::Array{Int64}:  Array with equation indices to add equiation interface for.\n\nKeywords\n\nworkingDir::String=pwd(): Working directory. Defaults to current working directory.\n\nReturns\n\nPath to generated FMU workingDir/<modelName>.interface.fmu.\n\nSee also profiling, generateFMU, generateTrainingData.\n\n\n\n\n\n","category":"function"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.generateFMU","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.generateFMU","text":"generateFMU(modelName, moFiles; options)\n\nGenerate 2.0 Model Exchange FMU for Modelica model using OMJulia.\n\nArguments\n\nmodelName::String:        Name of the Modelica model.\nmoFiles::Array{String}:   Path to the *.mo file(s) containing the model.\n\nKeywords\n\noptions::OMOptions:       Options for OpenModelica compiler.\n\nReturns\n\nPath to generated FMU workingDir/<modelName>.fmu.\n\nSee also OMOptions, addEqInterface2FMU, generateTrainingData.\n\n\n\n\n\n","category":"function"},{"location":"dataGen/#Structures","page":"Data Generation","title":"Structures","text":"","category":"section"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"DataGenOptions","category":"page"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.DataGenOptions","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.DataGenOptions","text":"DataGenOptions <: Any\n\nSettings for data generation.\n\nmethod::NonLinearSystemNeuralNetworkFMU.DataGenerationMethod: Method to generate data points. Allowed values: RandomMethod, RandomWalkMethod\nn::Integer: Number of data points to generate.\nnBatches::Integer: Number of batches to divide N into.\nnThreads::Integer: Number of threads to use in parallel\nappend::Bool: Append to already existing data\nclean::Bool: Clean up temp CSV files\n\nSee als RandomMethod, RandomWalkMethod.\n\n\n\n\n\n","category":"type"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"RandomMethod\nRandomWalkMethod","category":"page"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.RandomMethod","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.RandomMethod","text":"RandomMethod <: DataGenerationMethod\n\nRandom data generation using rand.\n\n\n\n\n\n","category":"type"},{"location":"dataGen/#NonLinearSystemNeuralNetworkFMU.RandomWalkMethod","page":"Data Generation","title":"NonLinearSystemNeuralNetworkFMU.RandomWalkMethod","text":"RandomWalkMethod <: DataGenerationMethod\n\nRandomized brownian-like motion data generation. Tries to stay within one solution in case the non-linear system is not unique solveable. Uses previous data point to generate close data point with previous solution as input to NLS solver.\n\ndelta::Float64: Step size of random walk.\n\n\n\n\n\n","category":"type"},{"location":"dataGen/#data_gen_example_id","page":"Data Generation","title":"Examples","text":"","category":"section"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"First we need to create a Model-Exchange 2.0 FMU with OpenModelica.","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"This can be done directly from OpenModelica or with generateFMU:","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"using NonLinearSystemNeuralNetworkFMU # hide\nmoFiles = [\"test/simpleLoop.mo\"]\noptions = OMOptions(workingDir = \"tempDir\")\n\nfmu = generateFMU(\"simpleLoop\",\n                  moFiles;\n                  options = options)","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"Next we need to add non-standard FMI function","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"fmi2Status myfmi2EvaluateEq(fmi2Component c, const size_t eqNumber)","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"that will call <modelname>_eqFunction_<eqIndex>(DATA* data, threadData_t *threadData) for all non-linear equations we want to generate data for.","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"Using addEqInterface2FMU this C code will be generated and added to the FMU.","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"interfaceFmu = addEqInterface2FMU(\"simpleLoop\",\n                                  fmu,\n                                  [14],\n                                  workingDir = \"tempDir\")","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"Now we can create evaluate equation 14 for random values and save the outputs to generate training data.","category":"page"},{"location":"dataGen/","page":"Data Generation","title":"Data Generation","text":"using CSV\nusing DataFrames\noptions=DataGenOptions(n=10, nThreads=1)\ngenerateTrainingData(interfaceFmu,\n                     \"tempDir\",\n                     \"simpleLoop_data.csv\",\n                     14,\n                     [\"s\", \"r\"],\n                     [0.0, 0.95],\n                     [1.5, 3.15],\n                     [\"y\"];\n                     options = options)\ndf =  DataFrame(CSV.File(\"simpleLoop_data.csv\"))","category":"page"},{"location":"main/#Main-Data-Generation-Routine","page":"Main","title":"Main Data Generation Routine","text":"","category":"section"},{"location":"main/","page":"Main","title":"Main","text":"To perform all needed steps for data generation the following functions have to be executed:","category":"page"},{"location":"main/","page":"Main","title":"Main","text":"profiling\ngenerateFMU\naddEqInterface2FMU\ngenerateTrainingData","category":"page"},{"location":"main/","page":"Main","title":"Main","text":"These functionalities are bundled in main.","category":"page"},{"location":"main/#Functions","page":"Main","title":"Functions","text":"","category":"section"},{"location":"main/","page":"Main","title":"Main","text":"main","category":"page"},{"location":"main/#NonLinearSystemNeuralNetworkFMU.main","page":"Main","title":"NonLinearSystemNeuralNetworkFMU.main","text":"main(modelName,\n     moFiles;\n     options=OMOptions(workingDir=joinpath(pwd(), modelName)),\n     dataGenOptions=DataGenOptions(method = RandomMethod(), n=1000, nBatches=Threads.nthreads()),\n     reuseArtifacts=false)\n\nMain routine to generate training data from Modelica file(s). Generate BSON artifacts and FMUs for each step. Artifacts can be re-used when restarting main routine to skip already performed stepps.\n\nWill perform profiling, min-max value compilation, FMU generation and data generation for all non-linear equation systems of modelName.\n\nArguments\n\nmodelName::String:      Name of Modelica model to simulate.\nmoFiles::Array{String}: Path to .mo file(s).\n\nKeywords\n\nomOptions::OMOptions:           Settings for OpenModelcia compiler.\ndataGenOptions::DataGenOptions  Settings for data generation.\nreuseArtifacts=false:           Use artifacts to skip already performed steps if true.\n\nReturns\n\ncsvFiles::Array{String}:              Array of generate CSV files with training data.\nfmu::String:                          Path to unmodified 2.0 ME FMU.\nprofilingInfo::Array{ProfilingInfo}:  Array of profiling information for each non-linear equation system.\n\nSee also profiling, minMaxValuesReSim, generateFMU, addEqInterface2FMU, generateTrainingData.\n\n\n\n\n\n","category":"function"},{"location":"main/#Example","page":"Main","title":"Example","text":"","category":"section"},{"location":"main/","page":"Main","title":"Main","text":"using NonLinearSystemNeuralNetworkFMU\nmodelName = \"simpleLoop\";\nmoFiles = [joinpath(\"test\",\"simpleLoop.mo\")];\nomOptions = OMOptions(workingDir=\"tempDir\")\ndataGenOptions = DataGenOptions(method=NonLinearSystemNeuralNetworkFMU.RandomMethod(),\n                                n=10,\n                                nBatches=2)\n\n(csvFiles, fmu, profilingInfo) = main(modelName,\n                                      moFiles;\n                                      omOptions=omOptions,\n                                      dataGenOptions=dataGenOptions,\n                                      reuseArtifacts=false)","category":"page"},{"location":"train/#Train-Machine-Learning-Surrogate","page":"ONNX Generation","title":"Train Machine Learning Surrogate","text":"","category":"section"},{"location":"train/","page":"ONNX Generation","title":"ONNX Generation","text":"With the generated training data it is possible to train a machine learning (ML) method of your choice, as log as it can be exported as an ONNX.","category":"page"},{"location":"train/","page":"ONNX Generation","title":"ONNX Generation","text":"This step has to be performed by the user.","category":"page"},{"location":"train/#Example","page":"ONNX Generation","title":"Example","text":"","category":"section"},{"location":"train/","page":"ONNX Generation","title":"ONNX Generation","text":"For a naive feed-forward neural network exported to ONNX see AnHeuermann/NaiveONNX.jl.","category":"page"},{"location":"train/","page":"ONNX Generation","title":"ONNX Generation","text":"using NonLinearSystemNeuralNetworkFMU # hide\nimport NaiveONNX\ntrainingData = \"simpleLoop_data.csv\"\nprofilingInfo = getProfilingInfo(\"simpleLoop.bson\")[1]\nonnxModel = \"eq_$(profilingInfo.eqInfo.id).onnx\" # Name of ONNX to generate\n\nmodel = NaiveONNX.trainONNX(trainingData,\n                            onnxModel,\n                            profilingInfo.usingVars,\n                            profilingInfo.iterationVariables;\n                            nepochs=10,\n                            losstol=1e-8)","category":"page"},{"location":"#NonLinearSystemNeuralNetworkFMU.jl","page":"Home","title":"NonLinearSystemNeuralNetworkFMU.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Generate Neural Networks to replace non-linear systems inside OpenModelica 2.0 FMUs.","category":"page"},{"location":"#Table-of-Contents","page":"Home","title":"Table of Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"  pages = [\n    \"Home\" => \"index.md\",\n    \"Main\" => \"main.md\",\n    \"Profiling\" => \"profiling.md\",\n    \"Data Generation\" => \"dataGen.md\",\n    \"ONNX Generation\" => \"train.md\",\n    \"Integrate ONNX\" => \"integrateONNX.md\"\n  ]","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package generates an FMU from a modelica file in 3 steps (+ 1 user step):","category":"page"},{"location":"","page":"Home","title":"Home","text":"Find non-linear equation systems to replace.\nSimulate and profile Modelica model with OpenModelica using OMJulia.jl.\nFind slowest equations below given threshold.\nFind depending variables specifying input and output for every non-linear equation system.\nFind min-max ranges for input variables by analyzing the simulation results.\nGenerate training data.\nGenerate 2.0 Model Exchange FMU with OpenModelica.\nAdd C interface to evaluate single non-linear equation system without evaluating anything else.\nRe-compile FMU.\nInitialize FMU using FMI.jl.\nGenerate training data for each equation system by calling new interface.\nCreate ONNX (performed by user).\nUse your favorite environment to create a trained Open Neural Network Exchange (ONNX) model.\nUse the generated training data to train artificial neural network.\nIntegrate ONNX into FMU.\nReplace equations with ONNX evaluation done by ONNX Runtime in generated C code.\nRe-compile FMU.\nEnvironment variable ORT_DIR has to be set and point to the ONNX runtime directory (with include/ and lib/ inside).","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl README.md for installation instructions.","category":"page"},{"location":"integrateONNX/#Include-ONNX-into-exported-FMU","page":"Integrate ONNX","title":"Include ONNX into exported FMU","text":"","category":"section"},{"location":"integrateONNX/","page":"Integrate ONNX","title":"Integrate ONNX","text":"After an ONNX is generated it can be compiled into the FMU.","category":"page"},{"location":"integrateONNX/","page":"Integrate ONNX","title":"Integrate ONNX","text":"warning: Warning\nThe FMU can't be compiled on Windows systems, because the ONNX Runtime is incompatible with the MSYS2 shell used by OpenModelica to compile the FMU.","category":"page"},{"location":"integrateONNX/#Functions","page":"Integrate ONNX","title":"Functions","text":"","category":"section"},{"location":"integrateONNX/","page":"Integrate ONNX","title":"Integrate ONNX","text":"buildWithOnnx","category":"page"},{"location":"integrateONNX/#NonLinearSystemNeuralNetworkFMU.buildWithOnnx","page":"Integrate ONNX","title":"NonLinearSystemNeuralNetworkFMU.buildWithOnnx","text":"buildWithOnnx(fmu, modelName, equations, onnxFiles; usePrevSol=false, tempDir=modelName*\"_onnx\")\n\nInclude ONNX into FMU and recompile to generate FMU with ONNX surrogates.\n\nArguments\n\nfmu::String:                        Path to FMU to extend with ONNX surrogates.\nmodelName::String:                  Name of model in FMU.\nequations::Array{ProfilingInfo}:    Profiling info for all equations to replace.\nonnxFiles::Array{String}:           Array of paths to ONNX surrogates.\n\nKeywords\n\nusePrevSol::Bool:                   ONNX uses previous solution as additional input.\ntempDir::String:                    Working directory\n\nReturns\n\nPath to ONNX FMU.\n\n\n\n\n\n","category":"function"},{"location":"integrateONNX/#Example","page":"Integrate ONNX","title":"Example","text":"","category":"section"},{"location":"integrateONNX/","page":"Integrate ONNX","title":"Integrate ONNX","text":"using NonLinearSystemNeuralNetworkFMU # hide\nrm(\"onnxTempDir\", recursive=true, force=true) # hide\nmodelName = \"simpleLoop\"\nfmu = joinpath(\"tempDir\", \"simpleLoop.interface.fmu\")\nprofilingInfo = getProfilingInfo(\"simpleLoop.bson\")[1:1]\nonnxFiles = [\"eq_14.onnx\"]\n\nbuildWithOnnx(fmu,\n              modelName,\n              profilingInfo,\n              onnxFiles,\n              tempDir = \"onnxTempDir\")","category":"page"},{"location":"integrateONNX/","page":"Integrate ONNX","title":"Integrate ONNX","text":"This FMU can now be simulated with most FMI importing tools.","category":"page"},{"location":"profiling/#Profiling-Modelica-Models","page":"Profiling","title":"Profiling Modelica Models","text":"","category":"section"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"The profiling functionalities of OpenModelica are used to decide if an equation is slow enough to be replaced by a surrogate.","category":"page"},{"location":"profiling/#Functions","page":"Profiling","title":"Functions","text":"","category":"section"},{"location":"profiling/#Profiling","page":"Profiling","title":"Profiling","text":"","category":"section"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"Simulate Modelica model to find slowest equations and what variables are used and what values these variables have during simulation.","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"profiling\nminMaxValuesReSim","category":"page"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.profiling","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.profiling","text":"profiling(modelName, moFiles; pathToOmc, workingDir, threshold = 0.03)\n\nFind equations of Modelica model that are slower then threashold.\n\nArguments\n\nmodelName::String:  Name of the Modelica model.\nmoFiles::Array{String}:   Path to the *.mo file(s) containing the model.\n\nKeywords\n\noptions::OMOptions:       Options for OpenModelica compiler.\nthreshold=0.01:           Slowest equations that need more then threshold of total simulation time.\nignoreInit::Bool=true:    Ignore equations from initialization system if true.\n\nReturns\n\nprofilingInfo::Vector{ProfilingInfo}: Profiling information with non-linear equation systems slower than threshold.\n\n\n\n\n\n","category":"function"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.minMaxValuesReSim","text":"minMaxValuesReSim(vars, modelName, moFiles; pathToOmc=\"\" workingDir=pwd())\n\n(Re-)simulate Modelica model and find miminum and maximum value each variable has during simulation.\n\nArguments\n\nvars::Array{String}:    Array of variables to get min-max values for.\nmodelName::String:      Name of Modelica model to simulate.\nmoFiles::Array{String}: Path to .mo file(s).\n\nKeywords\n\noptions::OMOptions:     Options for OpenModelica compiler.\n\nReturns\n\nmin::Array{Float64}: Minimum values for each variable listed in vars, minus some small epsilon.\nmax::Array{Float64}: Maximum values for each variable listed in vars, plus some small epsilon.\n\nSee also profiling.\n\n\n\n\n\n","category":"function"},{"location":"profiling/#Getting-Profiling","page":"Profiling","title":"Getting Profiling","text":"","category":"section"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"The main function will save profiling artifacts that can be loaded with the following functions.","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"getProfilingInfo\ngetUsingVars\ngetIterationVars\ngetInnerEquations\ngetMinMax","category":"page"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.getProfilingInfo","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.getProfilingInfo","text":"getProfilingInfo(bsonFile)\n\nRead ProfilingInfo array from binary JSON file.\n\n\n\n\n\n","category":"function"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.getUsingVars","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.getUsingVars","text":"getUsingVars(bsonFile, eqNumber)\n\nArguments\n\nbsonFile::String:  name of the binary JSON file\neqNumber::Int:  number of the slowest equation\n\nReturn:\n\nArray of used variables\n\n\n\n\n\n","category":"function"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.getIterationVars","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.getIterationVars","text":"getIterationVars(bsonFile, eqNumber)\n\nArguments\n\nbsonFile::String:  name of the binary JSON file\neqNumber::Int:  number of the slowest equation\n\nReturn:\n\nArray of iteration variables\n\n\n\n\n\n","category":"function"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.getInnerEquations","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.getInnerEquations","text":"getInnerEquations(bsonFile, eqNumber)\n\nArguments\n\nbsonFile::String:  name of the binary JSON file\neqNumber::Int:  number of the slowest equation\n\nReturn:\n\nArray of inner equation indices\n\n\n\n\n\n","category":"function"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.getMinMax","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.getMinMax","text":"getMinMax(bsonFile, eqNumber, inputArray)\n\nArguments\n\nbsonFile::String:  name of the binary JSON file\neqNumber::Int:  number of the slowest equation\ninputArray::Vector{String}: array of input variables as String\n\nReturn:\n\narray of the min and max values of each input from input array\n\n\n\n\n\ngetMinMax(bsonFile, eqNumber, inputArray)\n\nArguments\n\nbsonFile::String:  name of the binary JSON file\neqNumber::Int:  number of the slowest equation\ninputArray::Vector{Int}: array of input variables as Integers\n\nReturn:\n\narray of the min and max values of each input from input array\n\n\n\n\n\n","category":"function"},{"location":"profiling/#Structures","page":"Profiling","title":"Structures","text":"","category":"section"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"OMOptions\nProfilingInfo\nEqInfo","category":"page"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.OMOptions","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.OMOptions","text":"OMOptions <: Any\n\nSettings for profiling and simulating with the OpenModelica Compiler (OMC).\n\npathToOmc::String: Path to omc used for simulating the model.\nworkingDir::String: Working directory for omc. Defaults to the current directory.\noutputFormat::Union{Nothing, String}: Output format for result file. Can be \"mat\" or \"csv\".\nclean::Bool: Remove everything in workingDir when set to true.\ncommandLineOptions::String: Additional comannd line options for setCommandLineOptions.\n\n\n\n\n\n","category":"type"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.ProfilingInfo","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.ProfilingInfo","text":"ProfilingInfo <: Any\n\nProfiling information for single non-linear equation.\n\neqInfo::EqInfo: Non-linear equation\niterationVariables::Array{String}: Iteration (output) variables of non-linear system\ninnerEquations::Array{Int64}: Inner (torn) equations of non-linear system.\nusingVars::Array{String}: Used (input) variables of non-linear system.\nboundary::NonLinearSystemNeuralNetworkFMU.MinMaxBoundaryValues{Float64}: Minimum and maximum boundary values of usingVars.\n\n\n\n\n\n","category":"type"},{"location":"profiling/#NonLinearSystemNeuralNetworkFMU.EqInfo","page":"Profiling","title":"NonLinearSystemNeuralNetworkFMU.EqInfo","text":"EqInfo <: Any\n\nEquation info struct.\n\nid::Int64: Unique equation id\nncall::Int64: Number of calls during simulation\ntime::Float64: Total time [s] spend on evaluating this equation.\nmaxTime::Float64: Maximum time [s] needed for single evaluation of equation.\nfraction::Float64: Fraction of total simulation time spend on evaluating this equation.\n\n\n\n\n\n","category":"type"},{"location":"profiling/#Examples","page":"Profiling","title":"Examples","text":"","category":"section"},{"location":"profiling/#Find-Slowest-Non-linear-Equation-Systems","page":"Profiling","title":"Find Slowest Non-linear Equation Systems","text":"","category":"section"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"We have a Modelica model SimpleLoop, see test/simpleLoop.mo with some non-linear equation system","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"beginalign*\n  r^2 = x^2 + y^2 \n  rs  = x + y\nendalign*","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"We want to see how much simulation time is spend solving this equation. So let's start profiling:","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"using NonLinearSystemNeuralNetworkFMU\nmodelName = \"simpleLoop\";\nmoFiles = [joinpath(\"test\",\"simpleLoop.mo\")];\noptions = OMOptions(workingDir = \"tempDir\")\nprofilingInfo = profiling(modelName, moFiles; options=options, threshold=0)","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"We can see that non-linear equation system 14 is using variables s and r as input and has iteration variable y. x will be computed in the inner equation.","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"profilingInfo[1].usingVars\nprofilingInfo[1].iterationVariables","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"So we can see, that equations 14 is the slowest non-linear equation system. It is called 2512 times and needs around 15% of the total simulation time, in this case that is around 592 mu s.","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"During profiling function minMaxValuesReSim is called to re-simulate the Modelica model and read the simulation results to find the smallest and largest values for each given variable.","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"We can check them by looking into","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"profilingInfo[1].boundary.min\nprofilingInfo[1].boundary.min","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"It's possible to save and later load the profilingInfo in binary JSON format:","category":"page"},{"location":"profiling/","page":"Profiling","title":"Profiling","text":"using BSON\nBSON.@save \"simpleLoop.bson\" profilingInfo\ngetProfilingInfo(\"simpleLoop.bson\")","category":"page"}]
}
