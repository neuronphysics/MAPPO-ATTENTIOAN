from pettingzoo.utils.wrappers import BaseParallelWrapper
import numpy as np
import gym
from supersuit.utils.wrapper_chooser import WrapperChooser


class black_death_par(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.act_masks = False

    def _check_valid_for_black_death(self):
        for agent in self.agents:
            space = self.observation_space(agent)
            if type(space) == gym.spaces.Dict:
                # We have action_masks
                self.act_masks = True
                obs_space = space['observation']
                assert isinstance(
                obs_space, gym.spaces.Box
            ), f"observation sapces for black death must be Box spaces, is {obs_space}"
            else:
                assert isinstance(
                    space, gym.spaces.Box
                ), f"observation sapces for black death must be Box spaces, is {space}"

    def reset(self, seed=None):
        obss = self.env.reset(seed=seed)
        self.agents = self.env.agents[:]
        self._check_valid_for_black_death()
        if type(obss[self.agents[0]]) == dict:
            # This contains action_mask
            black_obs = {
                agent: {
                    'observation': np.zeros_like(self.observation_space(agent)['observation'].low),
                    'action_mask': np.zeros_like(self.observation_space(agent)['action_mask'].low)
                    }
                for agent in self.agents
                if agent not in obss
            }
        else:
            black_obs = {
                agent: np.zeros_like(self.observation_space(agent).low)
                for agent in self.agents
                if agent not in obss
            }
        return {**obss, **black_obs}

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, dones, infos = self.env.step(active_actions)
        agents = list(obss.keys())
        if self.act_masks:
            if len(agents) > 0:
                # This contains action_mask
                black_obs = {
                    agent: {
                        'observation': np.zeros_like(self.observation_space(agent)['observation'].low),
                        'action_mask': np.zeros_like(self.observation_space(agent)['action_mask'].low)
                        }
                    for agent in self.agents
                    if agent not in obss
                }
            else:
                # This contains action_mask
                black_obs = {
                    agent: {
                        'observation': np.zeros_like(self.observation_space(agent)['observation'].low),
                        'action_mask': np.zeros_like(self.observation_space(agent)['action_mask'].low)
                        }
                    for agent in self.agents
                }
        else:
            black_obs = {
                agent: np.zeros_like(self.observation_space(agent).low)
                for agent in self.agents
                if agent not in obss
            }
        black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        env_is_done = all(dones.values())
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}
        total_dones = {agent: env_is_done for agent in self.agents}
        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, total_dones, total_infos


black_death_v3 = WrapperChooser(parallel_wrapper=black_death_par)