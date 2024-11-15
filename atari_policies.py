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
        # Preprocess the input: convert to float and scale to [0, 1]
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
        # For discrete actions, use CategoricalDistribution
        self.action_dist = CategoricalDistribution(action_space.n)

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = AtariCNN(self.observation_space, self.features_dim)
        self.mlp_extractor = MCPPacmanModel(
            input_dim=self.features_dim,
            action_dim=self.action_space.n,
            num_primitives=self.num_primitives
        )

    def forward_actor(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        return self.mlp_extractor.forward_actor(features)

    def freeze_primitives(self):
        def zero_grad_hook(grad):
            return grad * 0
        
        # Freeze feature extractor
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        
        # Freeze primitives
        for primitive in self.mlp_extractor.primitives:
            for param in primitive.parameters():
                param.requires_grad = False
        
        # Keep gate trainable
        for param in self.mlp_extractor.gate.parameters():
            param.requires_grad = True

    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)
        observation = th.as_tensor(observation).float().to(self.device)
        with th.no_grad():
            features = self.extract_features(observation)
            weights = self.mlp_extractor.gate(features)
        return weights.cpu().numpy()