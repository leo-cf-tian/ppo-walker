from collections import namedtuple
from itertools import count
import gym
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import random
import time
import torch
from torch import nn, distributions, optim

Trajectory = namedtuple('Trajectory', ('observation', 'action', 'log_prob', 'reward', 'reward_to_go'))

class PPOAgent():
    def __init__(self, env: gym.Env):
        """
        Initialize the PPO agent
        """
        self.env = env
        self.set_hyperparams()
        
        self.critic = ValueApproximator(self.env.observation_space.shape[0])
        self.actor = PolicyApproximator(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        
    def set_hyperparams(self, batch_size=2000, num_minibatch=3, updates_per_iteration=5, learning_rate=0.005, grad_clip=5, discount_factor=0.99, covariance=0.5, entropy_coefficient=0.0005, clip=0.2):
        """
        Sets hyperparameters of the agent
        """
        self.gamma = discount_factor
        self.covariance = torch.diag(torch.full((self.env.action_space.shape[0],), covariance))
        self.epsilon = clip
        self.entropy_coefficient = entropy_coefficient
        
        self.batch_size = batch_size
        self.num_minibatch = num_minibatch
        self.updates_per_iteration = updates_per_iteration
        
        self.lr = learning_rate
        self.grad_clip = grad_clip
        
    def load_models(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt")))
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt")))
        
    def collect_batch(self, render):
        """
        Collects a batch of data by sampling
        """
        rollout: list[Trajectory] = []
        episodic_return = []
        
        while len(rollout) < self.batch_size:
            trajectory: list[Trajectory] = []
        
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float32)
        
            for t in count():
                action, log_prob = self.get_action(observation)
                new_observation, reward, terminated, truncated = self.env.step(action)
                
                if not torch.is_tensor(reward):
                    reward = torch.tensor(reward, dtype=torch.float32)

                trajectory.append(Trajectory(observation, action, log_prob, reward, 0))

                done = terminated or truncated

                observation = torch.tensor(new_observation, dtype=torch.float32).flatten()

                if render and len(rollout) == 0:
                    self.env.render()
                    
                if done:
                    break
            
            rewards = []
            reward_to_go = 0
            for timestep in reversed(trajectory):
                reward_to_go = (timestep.reward + reward_to_go * self.gamma).to(torch.float32)
                rewards.append(timestep.reward)
                rollout.append(timestep._replace(reward_to_go=reward_to_go))
                
            episodic_return.append(sum(rewards))
            
        return rollout, sum(episodic_return) / len(episodic_return)
    
    def get_action(self, observation: torch.Tensor, random = True) -> torch.Tensor:
        """
        Queries actor for the action using a gaussian policy with fixed covariance
        and calculates log probability for probability ratio later
        """
        mean_action = self.actor.forward(observation)
        
        dist = distributions.MultivariateNormal(mean_action, self.covariance)
        
        if (random):
            action = dist.sample()
        else:
            action = mean_action
            
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach()
    
    def train(self, iterations=1000, render=False, render_threshold=0):
        random.seed(time.time())
        
        avg_return = []
        timesteps = []
        
        plt.ion()
        figure, ax = plt.subplots()
        return_line, = ax.plot([], [], label="Return")
            
        for i in range(iterations):
            rollout, mean_return = self.collect_batch(render and i > render_threshold)
            rollout = Trajectory(*[torch.stack(list(t)) for t in zip(*rollout)])
            
            value = self.critic.forward(rollout.observation).squeeze()
            advantage = rollout.reward_to_go - value.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            
            actor_losses = []
            critic_losses = []
            
            indices = list(range(len(advantage)))
            
            for j in range(self.updates_per_iteration):
                random.shuffle(indices)
                for k in range(0, len(indices), len(indices) // self.num_minibatch):
                    actor_losses.append(self.update_actor(rollout.observation, rollout.action, rollout.log_prob, advantage))
                    critic_losses.append(self.update_critic(rollout.observation, rollout.reward_to_go))
                
            avg_return.append(mean_return)
            timesteps.append(len(rollout.reward) + timesteps[-1] if len(timesteps) > 0 else 0)
            
            return_line.set_xdata(timesteps)
            return_line.set_ydata(avg_return)
            ax.set_xlim([0, timesteps[-1]])
            ax.set_ylim([min(avg_return), max(avg_return)])
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.set_title(f"Iteration: {i + 1}")
            figure.canvas.draw()
            figure.canvas.flush_events()
                
    def run(self):
        while True:
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float32)
            
            for t in count():
                action, _ = self.get_action(observation)
                observation, _, terminated, truncated = self.env.step(action)
                observation = torch.tensor(observation, dtype=torch.float32).flatten()
                    
                self.env.render()
                time.sleep(0.001)

                if terminated or truncated:
                    break
                          
    def update_actor(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        advantage: torch.Tensor
    ):
        mean_action = self.actor.forward(observation)
        dist = distributions.MultivariateNormal(mean_action, self.covariance)
        log_prob_new = dist.log_prob(action)
        
        likelihood_ratio = torch.exp(log_prob_new - log_prob)
        
        surrogate_advantage = likelihood_ratio * advantage
        clipped_surrogate_advantage = torch.clamp(likelihood_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        
        loss = (-torch.min(surrogate_advantage, clipped_surrogate_advantage).mean())

        entropy_loss = -dist.entropy().mean() * self.entropy_coefficient
        
        loss = loss + entropy_loss
        
        self.actor_optim.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()
        
        return loss
        
    def update_critic(
        self, 
        observation: torch.Tensor,
        reward_to_go: torch.Tensor
    ):
        value = self.critic.forward(observation).squeeze()
        loss = nn.MSELoss()(value, reward_to_go)
    
        self.critic_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()
        
        return loss
        
        
class PolicyApproximator(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class ValueApproximator(nn.Module):
    def __init__(self, n_observations: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == '__main__':
    BASE_PATH = "./models"
    ENV_NAME = "MountainCarContinuous-v0"
    
    pretrained = False
    
    env = gym.make(ENV_NAME)
    agent = PPOAgent(env)
    
    if (pretrained):
        agent.load_models(os.path.join(BASE_PATH, ENV_NAME))
    
    agent.train(500, True, 0)
    
    torch.save(agent.actor.state_dict(), os.path.join(BASE_PATH, ENV_NAME, "actor.pt"))
    torch.save(agent.critic.state_dict(), os.path.join(BASE_PATH, ENV_NAME, "critic.pt"))
    
    # agent.run()