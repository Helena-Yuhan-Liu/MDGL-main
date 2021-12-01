#!/bin/sh
n_runs=5

sim_mode=${1?Error: no mode given}
echo "sim_mode ${sim_mode}"

if [ "$sim_mode" = "default" ]; then
    for i in $(seq $n_runs)
    do
         python3 AccumulatingCues_main.py --save_data=True --NP=False --comment="${sim_mode}_BPTT"
         python3 AccumulatingCues_main.py --save_data=True --NP=True --NP_mode=1 --comment="${sim_mode}_Eprop"
         python3 AccumulatingCues_main.py --save_data=True --NP=True --NP_mode=3 --comment="${sim_mode}_EI"         
    done
fi

