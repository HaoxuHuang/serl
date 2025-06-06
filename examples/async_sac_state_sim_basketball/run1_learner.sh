export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --learner_steps 750000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --checkpoint_period 10000 \
    --discount 0.999 \
    --checkpoint_path "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --load_checkpoint "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_2025-05-25_11-39-45/checkpoint_400000" \
    --debug # wandb is disabled when debug
