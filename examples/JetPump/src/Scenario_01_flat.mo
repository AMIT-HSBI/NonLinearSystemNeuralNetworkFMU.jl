package JetPumpTool "Static relations for sizing of jet pump geometry"
  extends Modelica.Icons.Package;

  package Components
    extends Modelica.Icons.BasesPackage;

    partial model EQS_SuctionFlow "Suction flow equation 85 to 90"
      input Modelica.Units.SI.AbsolutePressure p_0 "total pressure at inlet";
      input Modelica.Units.SI.Temperature T_0 "total temperature at inlet";
      input Modelica.Units.SI.AbsolutePressure p "static pressure at outlet";
      output Modelica.Units.SI.Temperature T = 338 "static temperature at outlet";
      output Modelica.Units.SI.MassFlowRate mflow = 0 "mass flow rate";
      Modelica.Units.SI.MachNumber Ma "Mach number at outlet";
      Modelica.Units.SI.VelocityOfSound c "velocity of sound at outlet";
      Modelica.Units.SI.Velocity v "fluid velocity at outlet";
      output Modelica.Units.SI.ImpulseFlowRate I = 0 "impulse at outlet";
      parameter Modelica.Units.SI.Area A = 7.5775215E-05 "area of outlet";
      parameter Real k = 1.4 "specific isentropic exponent";
      parameter Modelica.Units.SI.SpecificHeatCapacity R_s = 716.41 "specific gas constant";
    equation
      p_0/p = (1 + (k - 1)/2*Ma^2)^(k/(k - 1)) "eq. 85";
      T_0/T = (1 + (k - 1)/2*Ma^2) "eq. 86";
      v = R_s*mflow/A*T/p "eq. 87";
      c = sqrt(k*R_s*T) "eq. 88";
      Ma = v/c "eq. 89";
      I = v*mflow "eq. 91";
    end EQS_SuctionFlow;

    model SuctionFlow_p0_T0_mflow "p unkown"
      extends JetPumpTool.Components.EQS_SuctionFlow(p_0 = _p_0, T_0 = _T_0, mflow = _mflow, p = _p, T = _T, I = _I);
      Modelica.Blocks.Interfaces.RealInput _p_0 "total pressure at inlet = static pressure at inlet assuming v_0=0";
      Modelica.Blocks.Interfaces.RealInput _T_0 "total temperature at inlet";
      Modelica.Blocks.Interfaces.RealInput _mflow "mass flow rate";
      Modelica.Blocks.Interfaces.RealOutput _p "static pressure at outlet";
      Modelica.Blocks.Interfaces.RealOutput _T "static temperature at outlet";
      Modelica.Blocks.Interfaces.RealOutput _I "impulse at outlet";
    end SuctionFlow_p0_T0_mflow;
  end Components;

  package Scenarios "Operation scenarios usind time varying inputs"
    extends Modelica.Icons.ExamplesPackage;

    model Sim_SuctionFlow_p0_T0_mflow
      extends Modelica.Icons.Example;
      Components.SuctionFlow_p0_T0_mflow suctionFlow(p(start = 100000));
      Modelica.Blocks.Interaction.Show.RealValue realValue1(significantDigits = 4);
      Modelica.Blocks.Interaction.Show.RealValue realValue2(significantDigits = 4);
      Modelica.Blocks.Interaction.Show.RealValue realValue4(significantDigits = 4);
      Modelica.Blocks.Sources.Trapezoid p_0_suc_trapz(amplitude = 1e5, rising = 1, width = 1, falling = 1, period = 4, offset = 1.4e5, startTime = 0.1);
      Modelica.Blocks.Sources.Sine sine_T_0_mot1(amplitude = 35, f = 0.1, offset = 338);
      Modelica.Blocks.Sources.Trapezoid mflow_suc1(amplitude = 6e-3, rising = 1, width = 1, falling = 1, period = 4.4, offset = 5.0e-3);
    equation
      connect(suctionFlow._p, realValue1.numberPort);
      connect(suctionFlow._T, realValue2.numberPort);
      connect(suctionFlow._I, realValue4.numberPort);
      connect(p_0_suc_trapz.y, suctionFlow._p_0);
      connect(sine_T_0_mot1.y, suctionFlow._T_0);
      connect(mflow_suc1.y, suctionFlow._mflow);
      annotation(experiment(StartTime = 0, StopTime = 100, Tolerance = 1e-6, Interval = 0.1));
    end Sim_SuctionFlow_p0_T0_mflow;
  end Scenarios;
end JetPumpTool;

package ModelicaServices "ModelicaServices (OpenModelica implementation) - Models and functions used in the Modelica Standard Library requiring a tool specific implementation"
  extends Modelica.Icons.Package;

  package Machine "Machine dependent constants"
    extends Modelica.Icons.Package;
    final constant Real eps = 1e-15 "Biggest number such that 1.0 + eps = 1.0";
    final constant Real small = 1e-60 "Smallest number such that small and -small are representable on the machine";
    final constant Real inf = 1e60 "Biggest Real number such that inf and -inf are representable on the machine";
    final constant Integer Integer_inf = OpenModelica.Internal.Architecture.integerMax() "Biggest Integer number such that Integer_inf and -Integer_inf are representable on the machine";
  end Machine;
  annotation(version = "4.0.0", versionDate = "2020-06-04", dateModified = "2020-06-04 11:00:00Z");
end ModelicaServices;

package Modelica "Modelica Standard Library - Version 4.0.0"
  extends Modelica.Icons.Package;

  package Blocks "Library of basic input/output control blocks (continuous, discrete, logical, table blocks)"
    extends Modelica.Icons.Package;
    import Modelica.Units.SI;

    package Interaction "Library of user interaction blocks to input and to show variables in a diagram animation"
      extends Modelica.Icons.Package;

      package Show "Library of blocks to show variables in a diagram animation"
        extends Modelica.Icons.Package;

        block RealValue "Show Real value from numberPort or from number input field in diagram layer dynamically"
          parameter Boolean use_numberPort = true "= true, if numberPort enabled" annotation(Evaluate = true, HideResult = true);
          input Real number = 0.0 "Number to visualize if use_numberPort=false (time varying)" annotation(HideResult = true);
          parameter Integer significantDigits(min = 1) = 2 "Number of significant digits to be shown";
          Modelica.Blocks.Interfaces.RealInput numberPort if use_numberPort "Number to be shown in diagram layer if use_numberPort = true" annotation(HideResult = true);
          Modelica.Blocks.Interfaces.RealOutput showNumber;
        equation
          if use_numberPort then
            connect(numberPort, showNumber);
          else
            showNumber = number;
          end if;
        end RealValue;
      end Show;
    end Interaction;

    package Interfaces "Library of connectors and partial models for input/output blocks"
      extends Modelica.Icons.InterfacesPackage;
      connector RealInput = input Real "'input Real' as connector";
      connector RealOutput = output Real "'output Real' as connector";

      partial block SO "Single Output continuous control block"
        extends Modelica.Blocks.Icons.Block;
        RealOutput y "Connector of Real output signal";
      end SO;

      partial block SignalSource "Base class for continuous signal source"
        extends SO;
        parameter Real offset = 0 "Offset of output signal y";
        parameter SI.Time startTime = 0 "Output y = offset for time < startTime";
      end SignalSource;
    end Interfaces;

    package Sources "Library of signal source blocks generating Real, Integer and Boolean signals"
      import Modelica.Blocks.Interfaces;
      extends Modelica.Icons.SourcesPackage;

      block Sine "Generate sine signal"
        import Modelica.Constants.pi;
        parameter Real amplitude = 1 "Amplitude of sine wave";
        parameter SI.Frequency f(start = 1) "Frequency of sine wave";
        parameter SI.Angle phase = 0 "Phase of sine wave";
        extends Interfaces.SignalSource;
      equation
        y = offset + (if time < startTime then 0 else amplitude*Modelica.Math.sin(2*pi*f*(time - startTime) + phase));
      end Sine;

      block Trapezoid "Generate trapezoidal signal of type Real"
        parameter Real amplitude = 1 "Amplitude of trapezoid";
        parameter SI.Time rising(final min = 0) = 0 "Rising duration of trapezoid";
        parameter SI.Time width(final min = 0) = 0.5 "Width duration of trapezoid";
        parameter SI.Time falling(final min = 0) = 0 "Falling duration of trapezoid";
        parameter SI.Time period(final min = Modelica.Constants.small, start = 1) "Time for one period";
        parameter Integer nperiod = -1 "Number of periods (< 0 means infinite number of periods)";
        extends Interfaces.SignalSource;
      protected
        parameter SI.Time T_rising = rising "End time of rising phase within one period";
        parameter SI.Time T_width = T_rising + width "End time of width phase within one period";
        parameter SI.Time T_falling = T_width + falling "End time of falling phase within one period";
        SI.Time T_start "Start time of current period";
        Integer count "Period count";
      initial algorithm
        count := integer((time - startTime)/period);
        T_start := startTime + count*period;
      equation
        when integer((time - startTime)/period) > pre(count) then
          count = pre(count) + 1;
          T_start = time;
        end when;
        y = offset + (if (time < startTime or nperiod == 0 or (nperiod > 0 and count >= nperiod)) then 0 else if (time < T_start + T_rising) then amplitude*(time - T_start)/rising else if (time < T_start + T_width) then amplitude else if (time < T_start + T_falling) then amplitude*(T_start + T_falling - time)/falling else 0);
      end Trapezoid;
    end Sources;

    package Icons "Icons for Blocks"
      extends Modelica.Icons.IconsPackage;

      partial block Block "Basic graphical layout of input/output block" end Block;
    end Icons;
  end Blocks;

  package Math "Library of mathematical functions (e.g., sin, cos) and of functions operating on vectors and matrices"
    extends Modelica.Icons.Package;

    package Icons "Icons for Math"
      extends Modelica.Icons.IconsPackage;

      partial function AxisLeft "Basic icon for mathematical function with y-axis on left side" end AxisLeft;

      partial function AxisCenter "Basic icon for mathematical function with y-axis in the center" end AxisCenter;
    end Icons;

    function sin "Sine"
      extends Modelica.Math.Icons.AxisLeft;
      input Modelica.Units.SI.Angle u "Independent variable";
      output Real y "Dependent variable y=sin(u)";
      external "builtin" y = sin(u);
    end sin;

    function asin "Inverse sine (-1 <= u <= 1)"
      extends Modelica.Math.Icons.AxisCenter;
      input Real u "Independent variable";
      output Modelica.Units.SI.Angle y "Dependent variable y=asin(u)";
      external "builtin" y = asin(u);
    end asin;

    function exp "Exponential, base e"
      extends Modelica.Math.Icons.AxisCenter;
      input Real u "Independent variable";
      output Real y "Dependent variable y=exp(u)";
      external "builtin" y = exp(u);
    end exp;
  end Math;

  package Constants "Library of mathematical constants and constants of nature (e.g., pi, eps, R, sigma)"
    extends Modelica.Icons.Package;
    import Modelica.Units.SI;
    import Modelica.Units.NonSI;
    final constant Real pi = 2*Modelica.Math.asin(1.0);
    final constant Real small = ModelicaServices.Machine.small "Smallest number such that small and -small are representable on the machine";
    final constant SI.Velocity c = 299792458 "Speed of light in vacuum";
    final constant SI.ElectricCharge q = 1.602176634e-19 "Elementary charge";
    final constant Real h(final unit = "J.s") = 6.62607015e-34 "Planck constant";
    final constant Real k(final unit = "J/K") = 1.380649e-23 "Boltzmann constant";
    final constant Real N_A(final unit = "1/mol") = 6.02214076e23 "Avogadro constant";
    final constant Real mu_0(final unit = "N/A2") = 4*pi*1.00000000055e-7 "Magnetic constant";
  end Constants;

  package Icons "Library of icons"
    extends Icons.Package;

    partial package ExamplesPackage "Icon for packages containing runnable examples"
      extends Modelica.Icons.Package;
    end ExamplesPackage;

    partial model Example "Icon for runnable examples" end Example;

    partial package Package "Icon for standard packages" end Package;

    partial package BasesPackage "Icon for packages containing base classes"
      extends Modelica.Icons.Package;
    end BasesPackage;

    partial package InterfacesPackage "Icon for packages containing interfaces"
      extends Modelica.Icons.Package;
    end InterfacesPackage;

    partial package SourcesPackage "Icon for packages containing sources"
      extends Modelica.Icons.Package;
    end SourcesPackage;

    partial package IconsPackage "Icon for packages containing icons"
      extends Modelica.Icons.Package;
    end IconsPackage;
  end Icons;

  package Units "Library of type and unit definitions"
    extends Modelica.Icons.Package;

    package SI "Library of SI unit definitions"
      extends Modelica.Icons.Package;
      type Angle = Real(final quantity = "Angle", final unit = "rad", displayUnit = "deg");
      type Area = Real(final quantity = "Area", final unit = "m2");
      type Time = Real(final quantity = "Time", final unit = "s");
      type Velocity = Real(final quantity = "Velocity", final unit = "m/s");
      type Acceleration = Real(final quantity = "Acceleration", final unit = "m/s2");
      type Frequency = Real(final quantity = "Frequency", final unit = "Hz");
      type ImpulseFlowRate = Real(final quantity = "ImpulseFlowRate", final unit = "N");
      type Pressure = Real(final quantity = "Pressure", final unit = "Pa", displayUnit = "bar");
      type AbsolutePressure = Pressure(min = 0.0, nominal = 1e5);
      type MassFlowRate = Real(quantity = "MassFlowRate", final unit = "kg/s");
      type ThermodynamicTemperature = Real(final quantity = "ThermodynamicTemperature", final unit = "K", min = 0.0, start = 288.15, nominal = 300, displayUnit = "degC") "Absolute temperature (use type TemperatureDifference for relative temperatures)" annotation(absoluteValue = true);
      type Temperature = ThermodynamicTemperature;
      type SpecificHeatCapacity = Real(final quantity = "SpecificHeatCapacity", final unit = "J/(kg.K)");
      type ElectricCharge = Real(final quantity = "ElectricCharge", final unit = "C");
      type VelocityOfSound = Real(final quantity = "Velocity", final unit = "m/s");
      type FaradayConstant = Real(final quantity = "FaradayConstant", final unit = "C/mol");
      type MachNumber = Real(final quantity = "MachNumber", final unit = "1");
    end SI;

    package NonSI "Type definitions of non SI and other units"
      extends Modelica.Icons.Package;
      type Temperature_degC = Real(final quantity = "ThermodynamicTemperature", final unit = "degC") "Absolute temperature in degree Celsius (for relative temperature use Modelica.Units.SI.TemperatureDifference)" annotation(absoluteValue = true);
    end NonSI;
  end Units;
  annotation(version = "4.0.0", versionDate = "2020-06-04", dateModified = "2020-06-04 11:00:00Z");
end Modelica;

model Scenario_01_flat
  extends JetPumpTool.Scenarios.Sim_SuctionFlow_p0_T0_mflow;
 annotation(experiment(StartTime = 0, StopTime = 100, Tolerance = 1e-6, Interval = 0.1));
end Scenario_01_flat;
