export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python test.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --port 1145 \
    --random_steps 10 \
    --actor_steps 40000 \
    --load_checkpoint "/home/drl/Code/serl/examples/async_sac_state_sim_basketball/checkpoints/checkpoints_10-14-27/checkpoint_560000" \
    # --debug
