#!/bin/bash -e
verilator --timing tc.v -j 8 --build -Wno-INITIALDLY -Wno-WIDTHEXPAND --binary --trace-fst
./obj_dir/Vtc +DUMP
