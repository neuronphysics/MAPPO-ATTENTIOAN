import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
import imageio

def _t2n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, tuple):
        return tuple(tensor.detach().cpu().numpy() for tensor in x)
    else:
        return x

class MeltingpotRunner(Runner):
    def __init__(self, config):
        super(MeltingpotRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        print('num episodes to run (shared):', episodes) 

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            print('at 1')
            self.compute()
            print('at 2')
            train_infos = self.train()
            print('at 3')
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            print('at 4')

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "Meltingpot":
                    idv_rews = []
                    for agent_id in range(self.num_agents):
                        train_infos["average_episode_rewards"] = np.mean(self.buffer[agent_id].rewards) * self.episode_length
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            print('at 5')

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            print('at 6')

    def warmup(self):
        # reset env
        #if --n_rollout_threads 6 && --substrate_name "territory__rooms"
        obs = self.envs.reset()
        # replay buffer
        share_obs = []
        agent_obs = []
        for sublist in obs:
            for item in sublist:
                if item:
                    rgb_player=[]
                    arrays = []
                    for agent_id in range(self.num_agents):
                        player= f"player_{agent_id}"
                        if player in item:
                              rgb_player.append(item[player]['WORLD.RGB'])
                              arrays.append(item[player]['RGB'])
                              #player_i obs: (11, 11, 3), share_obs: (168, 168, 3)
                    result = np.stack(arrays)
                    image  = np.stack(rgb_player)
            share_obs.append(image)
            agent_obs.append(result)
        share_obs = np.array(share_obs)
        agent_obs = np.array(agent_obs)
        #share_obs shape: (6, 9, 168, 168, 3), agent obs shape: (6, 9, 11, 11, 3)
        # for agent_id in range(self.num_agents):
            #size of buffer share_obs (6, 168, 168, 3)--- obs (6, 11, 11, 3)
        self.buffer.share_obs[0] = share_obs[:,0,:,:,:].transpose(0, 2, 1, 3).copy()
        self.buffer.obs[0]       = agent_obs[:,0,:,:,:].copy()

        # # reset env
        # obs = self.envs.reset()

        # # replay buffer
        # if self.use_centralized_V:
        #     share_obs = obs.reshape(self.n_rollout_threads, -1)
        #     share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # else:
        #     share_obs = obs

        # print('share obs?', share_obs)

        # self.buffer.share_obs[0] = share_obs.copy()
        # self.buffer.obs[0] = obs.copy()


    # @torch.no_grad()
    # def collect(self, step):
    #     self.trainer.prep_rollout()
    #     value, action, action_log_prob, rnn_states, rnn_states_critic \
    #         = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
    #                         np.concatenate(self.buffer.obs[step]),
    #                         np.concatenate(self.buffer.rnn_states[step]),
    #                         np.concatenate(self.buffer.rnn_states_critic[step]),
    #                         np.concatenate(self.buffer.masks[step]))
    #     # [self.envs, agents, dim]
    #     values = np.array(np.split(_t2n(value), self.n_rollout_threads))
    #     actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
    #     action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
    #     rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
    #     rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
    #     # rearrange action
    #     if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
    #         for i in range(self.envs.action_space[0].shape):
    #             uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
    #             if i == 0:
    #                 actions_env = uc_actions_env
    #             else:
    #                 actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
    #     elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
    #         actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
    #     else:
    #         raise NotImplementedError

    #     return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # Convert tensors to numpy arrays
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        rnn_states = _t2n(rnn_states)
        rnn_states_critic = _t2n(rnn_states_critic)

        # Process actions based on action space type
        action_space_type = self.envs.action_space['player_0'].__class__.__name__
        if action_space_type == 'MultiDiscrete':
            actions_env = np.stack([np.eye(space.n)[actions[:, i]] for i, space in enumerate(self.envs.action_space.nvec)], axis=-1)
        elif action_space_type == 'Discrete':
            actions_env = np.eye(self.envs.action_space['player_0'].n)[actions]
        else:
            raise NotImplementedError(f"Action space type '{action_space_type}' not supported.")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env


    def insert(self, data):
        
        obs, rewards, done, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        print('len rnn states', len(rnn_states))
        print('rnn states', rnn_states)

        # Extract the boolean values for each player and convert to a boolean array -JUAN ADDED
        done  = np.array([player_dict[f'player_{0}'] for player_dict in done for i in range(self.num_agents)], dtype=np.bool_)        
        print("dones", done)
        print("rewards", np.array([player_dict[f'player_{i}'] for player_dict in rewards for i in range(self.num_agents)], dtype=np.float32))
        
        # rnn_states[done == True] = np.zeros(((done == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[done == True] = np.zeros(((done == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        for i in range(len(rnn_states)):
            print('rnn shape', rnn_states[i].shape)
            rnn_states[i][done] = np.zeros(((done == True).sum(), self.hidden_size), dtype=np.float32)
        for i in range(len(rnn_states_critic)):
            rnn_states_critic[i][done] = np.zeros(((done == True).sum(), self.hidden_size), dtype=np.float32)
            
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[done == True] = np.zeros(((done == True).sum(), 1), dtype=np.float32)

        share_obs = []
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self,total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_truncations, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                self.envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.envs.action_space[0].shape):
                        uc_actions_env = np.eye(self.envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, truncations, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)