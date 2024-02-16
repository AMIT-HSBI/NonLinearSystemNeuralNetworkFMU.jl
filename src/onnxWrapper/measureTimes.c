//
//
// Copyright (c) 2023 Andreas Heuermann
//
// This file is part of NonLinearSystemNeuralNetworkFMU.jl.
//
// NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
//
//
// ORT_ABORT_ON_ERROR taken from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//

#include "measureTimes.h"

void tic(struct timer* t) {
  gettimeofday(&(t->start), NULL);
}

double toc(struct timer* t) {
  double elapsedTime;
  gettimeofday(&(t->stop), NULL);
  elapsedTime = (t->stop.tv_sec - t->start.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t->stop.tv_usec - t->start.tv_usec) / 1000.0;   // us to ms

  return elapsedTime;
}
