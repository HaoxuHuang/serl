export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python bc.py "$@" \
    --learner \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --learner_steps 20001 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --checkpoint_period 5000 \
    --discount 0.999 \
    --checkpoint_path "/home/pjlab/serl/examples/async_sac_state_real_basketball/checkpoints/checkpoints_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --load_demo_path "/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/basketball_5x6_demos_2025-05-30_F" \
    --debug # wandb is disabled when debug
