# run_async_sac.ps1

# Disable preallocation and limit memory fraction for XLA client
$Env:XLA_PYTHON_CLIENT_PREALLOCATE = 'false'
$Env:XLA_PYTHON_CLIENT_MEM_FRACTION = '0.05'

# Invoke the Python script with passed-in arguments ($args)
python async_sac_state_sim.py @args `
    --learner `
    --env PandaPickCube-v0 `
    --exp_name serl_dev_sim_test `
    --seed 0 `
    --training_starts 1000 `
    --critic_actor_ratio 8 `
    --batch_size 256 `
    --debug  # wandb is disabled when debug
