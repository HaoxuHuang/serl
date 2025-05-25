export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --actor \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 0 \
    --actor_steps 7500000 \
    --render \
    --sleep_time 5 \
    --load_checkpoint "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_2025-05-25_11-39-45/checkpoint_400000" \
    --debug
