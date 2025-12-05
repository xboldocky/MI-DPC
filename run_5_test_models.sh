env="/home/desktop309/git/.venv/bin/python"

nohup taskset -c 10 $env -u _5_test_models.py > \
 test_models.log 2>&1  &