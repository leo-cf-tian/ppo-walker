from collections import namedtuple
from itertools import count
import gym
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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
        
    def set_hyperparams(self, batch_size=1000, learning_rate=0.005, grad_clip=5, discount_factor=0.99, covariance=0.5, clip=0.2):
        """
        Sets hyperparameters of the agent
        """
        self.gamma = discount_factor
        self.covariance = torch.diag(torch.full((self.env.action_space.shape[0],), covariance))
        self.epsilon = clip
        self.batch_size = batch_size
        self.lr = learning_rate
        self.grad_clip = grad_clip
        
    def collect_batch(self, render):
        """
        Collects a batch of data by sampling
        """
        rollout: list[Trajectory] = []
        episodic_return = []
        
        while len(rollout) < self.batch_size:
            trajectory: list[Trajectory] = []
        
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float)
        
            for t in count():
                action, log_prob = self.get_action(observation)
                new_observation, reward, terminated, truncated = self.env.step(action)

                trajectory.append(Trajectory(observation, action, log_prob, reward, 0))

                done = terminated or truncated

                observation = torch.tensor(new_observation, dtype=torch.float).flatten()

                if render and len(rollout) == 0:
                    self.env.render()
                    
                if done:
                    break
            
            returns = []
            reward_to_go = 0
            for timestep in reversed(trajectory):
                reward_to_go = timestep.reward + reward_to_go * self.gamma
                returns.append(timestep.reward)
                rollout.append(timestep._replace(reward_to_go=reward_to_go))
                
            episodic_return.append(sum(returns))
            
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
    
    def train(self, iterations=1000, updates_per_iteration=5, render=False):
        
        avg_return = []
        avg_actor_loss = []
        avg_critic_loss = []
        timesteps = []
        
        plt.ion()
        figure, ax = plt.subplots()
        return_line, = ax.plot([], [], label="Return")
        # actor_line, = ax.plot([], [], label="Actor Loss")
        # critic_line, = ax.plot([], [], label="Critic Loss")
            
        for i in range(iterations):
            rollout, mean_return = self.collect_batch(render and i > 400)
            rollout = Trajectory(*[torch.stack(list(t)) for t in zip(*rollout)])
            
            value = self.critic.forward(rollout.observation).squeeze()
            advantage = rollout.reward_to_go - value.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            
            actor_losses = []
            critic_losses = []
            
            for j in range(updates_per_iteration):
                actor_losses.append(self.update_actor(rollout, advantage))
                critic_losses.append(self.update_critic(rollout))
                
            avg_return.append(mean_return)
            avg_actor_loss.append((sum(actor_losses) / len(actor_losses)).item())
            avg_critic_loss.append((sum(critic_losses) / len(critic_losses)).item() / 5000)
            timesteps.append(len(rollout.reward) + timesteps[-1] if len(timesteps) > 0 else 0)
            
            return_line.set_xdata(timesteps)
            return_line.set_ydata(avg_return)
            # actor_line.set_xdata(timesteps)
            # actor_line.set_ydata(avg_actor_loss)
            # critic_line.set_xdata(timesteps)
            # critic_line.set_ydata(avg_critic_loss)
            ax.set_xlim([0, timesteps[-1]])
            ax.set_ylim([min(avg_return), max(avg_return)])
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Value')
            ax.legend()
            figure.canvas.draw()
            figure.canvas.flush_events()
                
                          
    def update_actor(self, rollout: Trajectory, advantage: torch.Tensor):
        mean_action = self.actor.forward(rollout.observation)
        dist = distributions.MultivariateNormal(mean_action, self.covariance)
        log_prob_new = dist.log_prob(rollout.action)
        
        likelihood_ratio = torch.exp(log_prob_new - rollout.log_prob)
        
        surrogate_advantage = likelihood_ratio * advantage
        clipped_surrogate_advantage = torch.clamp(likelihood_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        
        loss = (-torch.min(surrogate_advantage, clipped_surrogate_advantage).mean())
        
        self.actor_optim.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()
        
        return loss
        
    def update_critic(self, rollout: Trajectory):
        value = self.critic.forward(rollout.observation)
        loss = nn.MSELoss()(value, rollout.reward_to_go)
    
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
    env = gym.make("Pendulum-v0")
    agent = PPOAgent(env)
    agent.train(100000, 5, False)
    torch.save(agent.actor.state_dict(), "./models/actor.pt")
    torch.save(agent.critic.state_dict(), "./models/critic.pt")