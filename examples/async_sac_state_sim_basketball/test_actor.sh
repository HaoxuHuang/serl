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
    --load_checkpoint "/media/user5/Data/zzy/1/School/THU/courses/202502/202502.80470253-0.deep_reinforcement_learning/homework/project/main/serl/examples/async_sac_state_sim_basketball/checkpoints/BCSAC_checkpoint_740000/checkpoint_740000" \
    # --debug
