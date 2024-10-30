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