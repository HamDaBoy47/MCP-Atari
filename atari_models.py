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
        self.hidden_dim = hidden_dim
        
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
        
        # Each primitive now outputs latent actions that will be mapped to the actual action space
        self.primitives = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, hidden_dim)  # Output latent actions instead of direct action logits
            ) for _ in range(num_primitives)
        ])
        
        # Add an action mapping layer that converts latent actions to actual action logits
        self.action_map = nn.Linear(hidden_dim, action_dim)
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        for primitive in self.primitives:
            nn.init.orthogonal_(primitive[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.action_map.weight, gain=0.01)

    def forward_actor(self, features):
        weights = self.gate(features)  # Shape: (batch_size, num_primitives)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_primitives, 1)
        
        # Get primitive latent actions
        primitive_latents = torch.stack([
            primitive(features) for primitive in self.primitives
        ], dim=1)  # Shape: (batch_size, num_primitives, hidden_dim)
        
        # Multiplicative composition in latent space
        composed_latent = (weights * primitive_latents).sum(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Map to action logits
        action_logits = self.action_map(composed_latent)  # Shape: (batch_size, action_dim)
        
        return action_logits

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def transfer_from(self, source_model):
        """Transfer weights from a source model with different action dimension"""
        # Copy feature extractor
        self.feature_extractor.load_state_dict(source_model.feature_extractor.state_dict())
        
        # Copy gate network
        self.gate.load_state_dict(source_model.gate.state_dict())
        
        # Copy primitive networks (they output to same latent dimension)
        self.primitives.load_state_dict(source_model.primitives.state_dict())
        
        # Initialize new action mapping layer (this will be trained during transfer)
        # Keep existing initialization for action_map