export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 10 \
    --learner_steps 750000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --port 1145 \
    --load_checkpoint "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_10-07-42" \
    --debug # wandb is disabled when debug
