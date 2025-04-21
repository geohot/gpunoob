#!/bin/bash -e
verilator --cc --timing --exe tc.v --build -Wno-INITIALDLY --binary
./obj_dir/Vtc
