export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python checkpoints_logger.py "$@" \
    --learner \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --load_offline_data \
    --data_store_path "/home/drl/Code/serl/examples/async_sac_state_real_basketball/demos/basketball_5x6_demos_2025-06-13_headless" \
    --debug # wandb is disabled when debug