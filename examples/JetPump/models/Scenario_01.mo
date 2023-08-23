model Scenario_01
  extends JetPumpTool.Scenarios.Sim_SuctionFlow_p0_T0_mflow;
  annotation(
    uses(JetPumpTool),
    experiment(StartTime = 0, StopTime = 100, Tolerance = 1e-6, Interval = 0.1));
end Scenario_01;