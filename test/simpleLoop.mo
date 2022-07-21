//
// Copyright (c) 2022 Philip Hannebohm
// Licensed under the MIT license. See LICENSE.md file in the project root for details.
//

model simpleLoop
  Real r(min = 0);
  Real s(min = -sqrt(2), max = sqrt(2));
  Real x(start=1.0), y(start=0.5);
  Real x_ref, y_ref;
equation
  r = 1+time;
  s = sqrt((2-time)*0.9);

  r^2 = x^2 + y^2;
  r*s = x + y;

  x_ref = r * (s/2 + sqrt(1/2-s^2/4));
  y_ref = r*s - x_ref;
  annotation(experiment(StartTime=0, StopTime=2));
end simpleLoop;
