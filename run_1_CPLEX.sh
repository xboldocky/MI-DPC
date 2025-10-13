# Run CPLEX using single thread
env="/home/desktop309/git/.venv/bin/python"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nohup $env _1_CPLEX.py > CPLEX_output.log 2>&1 &