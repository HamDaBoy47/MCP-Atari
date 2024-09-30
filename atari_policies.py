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

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = AtariCNN(self.observation_space, self.features_dim)
        self.mlp_extractor = MCPPacmanModel(
            input_dim=self.features_dim,
            action_dim=self.action_space.n,
            num_primitives=self.num_primitives
        )

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
        return self.action_dist.proba_distribution(latent_pi).get_actions(deterministic=deterministic)

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
        """
        Preprocess the observation if needed and extract features.
        """
        obs = obs.float()  # Ensure input is float
        return self.features_extractor(obs)
    
class MCPTransferPolicy(MCPAtariPolicy):
    def __init__(self, *args, **kwargs):
        super(MCPTransferPolicy, self).__init__(*args, **kwargs)
        
    def freeze_primitives(self):
        for param in self.mlp_extractor.primitives.parameters():
            param.requires_grad = False
        
    def unfreeze_gate(self):
        for param in self.mlp_extractor.gate.parameters():
            param.requires_grad = True

    def adapt_to_new_action_space(self, new_action_space, old_weights):
        # Update the primitives
        new_primitives = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, new_action_space)
            ) for _ in range(self.num_primitives)
        ])
        
        # Copy weights where possible
        for i, (old_prim, new_prim) in enumerate(zip(self.mlp_extractor.primitives, new_primitives)):
            new_prim[0].weight.data = old_prim[0].weight.data
            new_prim[0].bias.data = old_prim[0].bias.data
            new_prim[1].weight.data = old_prim[1].weight.data
            new_prim[1].bias.data = old_prim[1].bias.data
            # Initialize the new output layer
            nn.init.orthogonal_(new_prim[2].weight.data, gain=0.01)
            nn.init.constant_(new_prim[2].bias.data, 0.0)
        
        self.mlp_extractor.primitives = new_primitives
        
        # Update the gate
        new_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, new_action_space),
            nn.Sigmoid()
        )
        
        # Copy weights where possible
        new_gate[0].weight.data = self.mlp_extractor.gate[0].weight.data
        new_gate[0].bias.data = self.mlp_extractor.gate[0].bias.data
        new_gate[1].weight.data = self.mlp_extractor.gate[1].weight.data
        new_gate[1].bias.data = self.mlp_extractor.gate[1].bias.data
        # Initialize the new output layer
        nn.init.orthogonal_(new_gate[2].weight.data, gain=0.01)
        nn.init.constant_(new_gate[2].bias.data, 0.0)
        
        self.mlp_extractor.gate = new_gate
        
        # Update the action distribution
        self.action_dist = CategoricalDistribution(new_action_space)
        
        # Update the action_net (which is actually just an identity function for categorical)
        self.action_net = nn.Identity()

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(action_logits=latent_pi)

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