model SoftStarter
  extends Modelica.Electrical.PowerConverters.Examples.ACAC.SoftStarter
  annotation(
    experiment(
      StopTime=10,
      Interval=0.0001,
      Tolerance=1e-06),
    uses(Modelica(version="4.0.0")));
end SoftStarter;
