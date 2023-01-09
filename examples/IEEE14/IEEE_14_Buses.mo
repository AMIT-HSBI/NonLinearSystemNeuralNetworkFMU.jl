model IEEE_14_Buses
  extends OpenIPSL.Examples.IEEE14.IEEE_14_Buses;
  annotation(
    uses(
      OpenIPSL(version="3.0.1")),
    experiment(
      StopTime=10,
      Interval=0.001,
      Tolerance=1e-06),
    __OpenModelica_commandLineOptions = "--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls");
end IEEE_14_Buses;
