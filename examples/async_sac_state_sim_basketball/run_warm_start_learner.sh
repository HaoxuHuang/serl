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
    --data_store_path "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/buffer" \
    --checkpoint_path "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --offline_data_steps 400000 \
    # --debug # wandb is disabled when debug