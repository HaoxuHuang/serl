export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python bc_actor.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --port 1145 \
    --random_steps 10 \
    --actor_steps 1000 \
    --save_demo_path "/home/pjlab/serl/examples/async_sac_state_sim_basketball/demos/demos_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --load_checkpoint "/home/pjlab/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_2025-05-23_16-03-25/checkpoint_170000" \
    # --debug
