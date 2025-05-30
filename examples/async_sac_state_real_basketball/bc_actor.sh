export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python bc.py "$@" \
    --actor \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 10 \
    --actor_steps 7500000 \
    --render \
    # --sleep_time 5 \
    # --debug
