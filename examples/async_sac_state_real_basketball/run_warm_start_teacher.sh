export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python /home/pjlab/serl/examples/async_sac_state_real_basketball/async_sac_with_offline_warm_start.py "$@" \
    --actor \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 0 \
    --actor_steps 3000 \
    --teacher \
    --load_checkpoint "/home/pjlab/serl/examples/async_sac_state_real_basketball/checkpoints/checkpoints_2025-05-30_22-09-06/checkpoint_20000" \
    # --data_store_path "/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/buffer" \
    # --sleep_time 2
    # --debug
