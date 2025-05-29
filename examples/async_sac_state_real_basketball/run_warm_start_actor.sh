export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_with_offline_warm_start.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 10 \
    --actor_steps 4000000 \
    --sleep_time 5 \
    --port 3109 \
    # --debug
