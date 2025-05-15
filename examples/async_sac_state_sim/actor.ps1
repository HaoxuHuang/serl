# run_async_sac_actor.ps1

# Configure XLA Python client environment
$Env:XLA_PYTHON_CLIENT_PREALLOCATE  = 'false'
$Env:XLA_PYTHON_CLIENT_MEM_FRACTION = '0.05'

# Launch the Python script with splatted arguments
python async_sac_state_sim.py @args `
    --actor `
    --render `
    --env PandaPickCube-v0 `
    --exp_name serl_dev_sim_test `
    --seed 0 `
    --random_steps 1000 `
    --debug
