import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from atari_models import MCPPacmanModel

class AtariCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(AtariCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))

class MCPAtariPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        num_primitives: int = 8,
        features_dim: int = 512,
        *args,
        **kwargs,
    ):
        self.num_primitives = num_primitives
        self.features_dim = features_dim
        super(MCPAtariPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = AtariCNN(self.observation_space, self.features_dim)
        self.mlp_extractor = MCPPacmanModel(
            input_dim=self.features_dim,
            action_dim=self.action_space.n,
            num_primitives=self.num_primitives
        )

    def freeze_primitives(self):
        for primitive in self.mlp_extractor.primitives:
            for param in primitive.parameters():
                param.requires_grad = False

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, value = self.mlp_extractor(features)
        distribution = self.action_dist.proba_distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(observation)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self.action_dist.proba_distribution(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        latent_pi, value = self.mlp_extractor(features)
        distribution = self.action_dist.proba_distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value, log_prob, entropy

    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)
        observation = th.as_tensor(observation).float().to(self.device)
        with th.no_grad():
            features = self.extract_features(observation)
            weights = self.mlp_extractor.gate(features)
        return weights.cpu().numpy()

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.float()
        return self.features_extractor(obs)

from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def main():
    # Specify render_mode when creating the environment
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')

    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=512),
        num_primitives=8
    )

    model = PPO(MCPAtariPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=100)

    # Test the trained model
    obs, _ = env.reset()
    
    plt.figure(figsize=(8,8))
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the game
        img = env.render()
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.01)  # Small pause to update display
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    plt.close()

if __name__ == "__main__":
    main()