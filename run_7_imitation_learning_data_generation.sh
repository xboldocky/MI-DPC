# env="/home/desktop309/git/.venv/bin/python"
# # solver=gurobi
# solver=cplex
# nohup taskset -c 20 $env _7_imitation_learning_data_generation.py > \
#  imitation_learning_data_generation_output_$solver.log 2>&1 --solver $solver &


#!/usr/bin/env bash

env="/home/desktop309/git/.venv/bin/python"
# solver=gurobi
solver=cplex

# CPUs to use (one per nsteps)
cpus=(10 11 12 13 14 15 16)

# Corresponding nsteps values
nsteps_list=(10 15 20 25 30 35 40)

for i in "${!nsteps_list[@]}"; do
    cpu="${cpus[$i]}"
    nsteps="${nsteps_list[$i]}"

    log="imitation_learning_data_generation_output_${solver}_nsteps${nsteps}.log"

    echo "Launching --nsteps ${nsteps} on CPU ${cpu}, logging to ${log}"

    nohup taskset -c "${cpu}" "${env}" -u _7_imitation_learning_data_generation.py \
        --solver "${solver}" \
        --nsteps "${nsteps}" \
        > "${log}" 2>&1 &
done
