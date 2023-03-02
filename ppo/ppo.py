import gym
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np

torch.autograd.set_detect_anomaly(True)

from model import FeedForwardNN


class PPO:

    def __init__(self, env: gym.Env):
        # Initialize hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]  # action space is continuous
        # self.act_dim = self.env.action_space.n  # action space is discrete

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Initialize actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create the covariance matrix for get_action
        self.cov_mat = torch.diag(torch.full(size=(self.act_dim,), fill_value=0.5))

    def learn(self, total_timesteps: int):
        t_so_far = 0

        while t_so_far < total_timesteps:   # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate $V_{\phi, k}$
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            # Calculate and normalize advantages
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # We do 5 updates after we retrieve 1 batch of training data
            for _ in range(self.n_updates_per_iteration):
                V, cur_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(cur_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Backward prop for actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate critic loss
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Backward prop for critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            t_so_far += np.sum(batch_lens)

    def rollout(self):
        # Batch data, one batch contains several episodes
        batch_obs = []  # observations, list
        batch_acts = []  # actions, list
        batch_log_probs = []  # log probs of actions, list
        batch_rewards = []  # rewards, list of lists
        batch_lens = []  # episode lengths, list of lists

        t = 0
        while t < self.timesteps_per_batch:
            print(t)
            # play one episode
            ep_rewards = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                # Act and step
                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                # Collect batch information
                batch_acts.append(action)
                ep_rewards.append(reward)
                batch_log_probs.append(log_prob)

                if done:
                    break
            # Collect episodic length and total rewards
            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        # Transform the batch data into tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        # ALG STEP 4
        batch_rtgs = torch.tensor(np.array(self.compute_rtgs(batch_rewards)), dtype=torch.float)
        print(f'finish making one batch, it contains {len(batch_lens)} episodes')
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        """
        Query the actor, retrieve action to take and the log prob of
        taking current action

        :param obs:
        :return:
        """
        # Query the actor network for action mean
        mean = self.actor(obs)

        # Create the sampling distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        # Calculate the log probabilities of batch actions using
        # most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()  # reshape (N, 1) to (N,)
        return V, log_probs

    def compute_rtgs(self, batch_rewards):
        """
        Compute reward-to-go

        :param batch_rewards: list of episode rewards
        :return:
        """
        batch_rtgs = []
        # Iterate each episode
        for ep_rewards in reversed(batch_rewards):
            discounted_return = 0
            for reward in reversed(ep_rewards):
                discounted_return = reward + discounted_return * self.gamma
                batch_rtgs.insert(0, discounted_return)  # insert at head(remember the order is reversed here)
        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 1000   # timesteps per batch
        self.max_timesteps_per_episode = 400  # timesteps per episode
        self.n_updates_per_iteration = 5  #
        self.gamma = 0.95  # discount factor
        self.lr = 0.005  # learning rate
        self.clip = 0.2  # clip threshold in ppo


env = gym.make('Pendulum-v0')
model = PPO(env)
model.learn(2000)
