export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_with_offline_rl.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 0 \
    --actor_steps 2000 \
    --teacher \
    --data_store_path "/media/user5/Data/zzy/1/School/THU/courses/202502/202502.80470253-0.deep_reinforcement_learning/homework/project/main/serl/examples/async_sac_state_sim_basketball/checkpoints/buffer/" \
    --load_checkpoint "/media/user5/Data/zzy/1/School/THU/courses/202502/202502.80470253-0.deep_reinforcement_learning/homework/project/main/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_2025-05-22_00-31-37"
    # --sleep_time 2
    # --debug
