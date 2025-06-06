export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_with_offline_warm_start.py "$@" \
    --learner \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --learner_steps 750000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --checkpoint_period 10000 \
    --load_offline_data \
    --data_store_path "/media/user5/Data/zzy/1/School/THU/courses/202502/202502.80470253-0.deep_reinforcement_learning/homework/project/main/serl/examples/async_sac_state_real_basketball/demos/basketball_1+5+5+5_demos_2025-05-26.pkl" \
    --checkpoint_path "/media/user5/Data/zzy/1/School/THU/courses/202502/202502.80470253-0.deep_reinforcement_learning/homework/project/main/serl/examples/async_sac_state_real_basketball/checkpoints/checkpoint_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --offline_decay_start 250000 \
    --offline_decay_steps 250000 \
    --offline_ratio 0.5 \
    --port 3109 \
    # --debug # wandb is disabled when debug