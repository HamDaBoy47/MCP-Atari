import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MCPPacmanModel(nn.Module):
    def __init__(self, input_dim, action_dim, num_primitives=8, hidden_dim=512):
        super(MCPPacmanModel, self).__init__()
        
        self.latent_dim_pi = action_dim  # Add this line
        self.latent_dim_vf = 1  # Add this line
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_primitives),
            nn.Sigmoid()
        )
        
        self.primitives = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            ) for _ in range(num_primitives)
        ])
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward_actor(self, features):
        weights = self.gate(features)
        
        primitive_outputs = torch.stack([primitive(features) for primitive in self.primitives], dim=1)
        composed_output = (weights.unsqueeze(-1) * primitive_outputs).sum(dim=1)
        
        return composed_output

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

def preprocess_observation(observation):
    """Preprocess the observation from gym to PyTorch tensor."""
    return torch.from_numpy(observation).float().flatten().unsqueeze(0)

# Example usage
if __name__ == "__main__":
    import gymnasium as gym
    
    # Create the environment
    env = gym.make('ALE/MsPacman-v5')
    
    # Get the number of actions from the environment
    num_actions = env.action_space.n
    
    # Calculate input dimension
    input_dim = np.prod(env.observation_space.shape)
    
    # Create the model
    model = MCPPacmanModel(input_dim=input_dim, action_dim=num_actions)
    
    # Get an observation from the environment
    observation, _ = env.reset()
    
    # Preprocess the observation
    state = preprocess_observation(observation)
    
    # Forward pass through the model
    action_probs, value = model(state)
    
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Value shape: {value.shape}")

