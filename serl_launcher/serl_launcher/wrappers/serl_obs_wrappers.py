import gym
from gym.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        if "images" in self.env.observation_space:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.env.observation_space["state"]),
                    **(self.env.observation_space["images"]),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.env.observation_space["state"]),
                }
            )

    def observation(self, obs):
        if "images" in obs:
            obs = {
                "state": flatten(self.env.observation_space["state"], obs["state"]),
                **(obs["images"]),
            }
        else:
            obs = {
                "state": flatten(self.env.observation_space["state"], obs["state"]),
            }
        return obs


class SERLObsWrapper2(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        if "images" in self.env.observation_space:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.env.observation_space["state"]),
                    **(self.env.observation_space["images"]),
                }
            )
        else:
            self.observation_space = flatten_space(self.env.observation_space)

    def observation(self, obs):
        if "images" in obs:
            obs = {
                "state": flatten(self.env.observation_space["state"], obs["state"]),
                **(obs["images"]),
            }
        else:
            obs = flatten(self.env.observation_space, obs)
        return obs
