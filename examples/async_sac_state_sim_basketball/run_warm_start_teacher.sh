export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_with_offline_warm_start.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 1000 \
    --actor_steps 3000 \
    --teacher \
    --data_store_path "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/buffer/" \
    --load_checkpoint "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_10-14-27/checkpoint_560000"
    # --sleep_time 2
    # --debug
