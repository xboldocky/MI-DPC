# Run CPLEX using single thread
# env="/home/desktop309/git/.venv/bin/python"
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
# nohup taskset -c $ic+10 $env _1_CPLEX.py -ic $ic > CPLEX_output.log 2>&1 &



#!/usr/bin/env bash

env="/home/desktop309/git/.venv/bin/python"

for ic in {0..19}; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nohup taskset -c $((ic+10)) "$env" _1_CPLEX.py -ic "$ic" \
    > "CPLEX_inference_data/logs/CPLEX_output_ic${ic}.log" 2>&1 &
done
