env="/home/desktop309/git/.venv/bin/python"
# solver=gurobi
solver=cplex
nohup $env _7_imitation_learning_data_generation.py > \
 imitation_learning_data_generation_output_$solver.log 2>&1 --solver $solver &