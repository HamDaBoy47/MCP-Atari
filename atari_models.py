# atari_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MCPPacmanModel(nn.Module):
    def __init__(self, input_dim, action_dim, num_primitives=8, hidden_dim=512):
        super(MCPPacmanModel, self).__init__()
        
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = 1
        self.action_dim = action_dim
        self.num_primitives = num_primitives
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Gate network outputs weights for combining primitives
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_primitives),
            nn.Sigmoid()
        )
        
        # Each primitive outputs logits for discrete actions
        self.primitives = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)  # Logits for discrete actions
            ) for _ in range(num_primitives)
        ])
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        for primitive in self.primitives:
            # Initialize the final layer with smaller weights to start
            nn.init.orthogonal_(primitive[-1].weight, gain=0.01)

    def forward_actor(self, features):
        weights = self.gate(features)  # Shape: (batch_size, num_primitives)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_primitives, 1)
        
        # Get primitive logits
        primitive_logits = torch.stack([
            primitive(features) for primitive in self.primitives
        ], dim=1)  # Shape: (batch_size, num_primitives, action_dim)
        
        # Multiplicative composition first
        composed_output = (weights * primitive_logits).sum(dim=1)  # Shape: (batch_size, action_dim)
        
        # Then convert to probabilities
        return composed_output

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)