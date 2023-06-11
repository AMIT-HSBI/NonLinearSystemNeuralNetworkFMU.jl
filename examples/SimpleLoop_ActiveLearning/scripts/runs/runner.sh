#! /usr/bin/env bash

for N in 600 800 1000 1200 1400
do
  for p in 0.0 0.25 0.5 0.75 1.0
  do
    julia run_${N}_${p}.jl > run_${N}_${p}.log 2>&1 &
  done
done