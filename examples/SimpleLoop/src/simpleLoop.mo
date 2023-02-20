//
// Copyright (c) 2022 Philip Hannebohm
//
// This file is part of NonLinearSystemNeuralNetworkFMU.jl.
//
// NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
//

model simpleLoop
  Real r(min = 0);
  Real s(min = -sqrt(2), max = sqrt(2));
  Real x(start=1.0), y(start=0.5);
  parameter Real b = -0.5;
  Real x_ref, y_ref;
equation
  r = 1+time;
  s = sqrt((2-time)*0.9);

  r^2 = x^2 + y^2;
  r*s + b = x + y;

  x_ref = 0.5 * ( -sqrt(-b^2 - 2*b*r*s - r^2*(s^2-2))+b+r*s);
  y_ref = r*s + b - x_ref;
  annotation(experiment(StartTime=0, StopTime=2));
end simpleLoop;
