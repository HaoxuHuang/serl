import pickle
import numpy

with open("/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/basketball_5x6_demos_2025-05-30", "rb") as f:
    d=pickle.load(f)

for transition in d:
    assert "next_observations" in transition
    assert "observations" in transition
    assert "actions" in transition
    assert "rewards" in transition
    assert "dones" in transition

# print(d)


