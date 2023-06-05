//
//
// Copyright (c) 2023 Andreas Heuermann
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
//
// ORT_ABORT_ON_ERROR taken from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//

#ifndef MEASURE_TIMES_H
#define MEASURE_TIMES_H

#include <stdio.h>
#include <sys/time.h>

struct timer {
  struct timeval start;
  struct timeval stop;
};

void tic(struct timer* t);
double toc(struct timer* t);

#endif // MEASURE_TIMES_H
