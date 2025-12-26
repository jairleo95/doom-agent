"""
Neural network models for Dreamer V3.

Contains all the neural network components:
- Encoder: CNN for encoding observations
- Decoder: CNN for reconstructing observations
- RSSM: Recurrent State-Space Model (world model core)
- RewardPredictor: Predicts rewards from latent states
- ContinuePredictor: Predicts episode continuation
- Actor: Policy network
- Critic: Value network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN Encoder for observations."""
    
    def __init__(self, obs_shape=(1, 120, 160), embed_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            conv_out = self.conv(dummy)
            self.flat_size = conv_out.shape[1]
        
        self.fc = nn.Linear(self.flat_size, embed_dim)
        
    def forward(self, obs):
        x = self.conv(obs)
        return self.fc(x)


class Decoder(nn.Module):
    """CNN Decoder for reconstructing observations."""
    
    def __init__(self, state_dim=512, obs_shape=(1, 120, 160)):
        super().__init__()
        self.obs_shape = obs_shape
        
        # Calculate initial spatial size
        self.init_h = obs_shape[1] // 16
        self.init_w = obs_shape[2] // 16
        
        self.fc = nn.Linear(state_dim, 256 * self.init_h * self.init_w)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], 4, stride=2, padding=1),
        )
        
    def forward(self, state):
        x = self.fc(state)
        x = x.view(-1, 256, self.init_h, self.init_w)
        return self.deconv(x)


class RSSM(nn.Module):
    """Recurrent State-Space Model - Core of Dreamer V3."""
    
    def __init__(self, action_dim=7, embed_dim=1024, hidden_dim=512, 
                 stochastic_dim=32, discrete_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.discrete_dim = discrete_dim
        
        # Recurrent model: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(
            stochastic_dim * discrete_dim + action_dim, 
            hidden_dim
        )
        
        # Prior: p(s_t | h_t)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, stochastic_dim * discrete_dim)
        )
        
        # Posterior: q(s_t | h_t, o_t)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, stochastic_dim * discrete_dim)
        )
        
    def forward(self, embed, action, hidden=None, stochastic=None):
        """
        Args:
            embed: Encoded observation [batch, embed_dim]
            action: Action taken [batch, action_dim]
            hidden: Previous hidden state [batch, hidden_dim]
            stochastic: Previous stochastic state [batch, stochastic_dim * discrete_dim]
        """
        batch_size = embed.shape[0]
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=embed.device)
        if stochastic is None:
            stochastic = torch.zeros(
                batch_size, self.stochastic_dim * self.discrete_dim, 
                device=embed.device
            )
        
        # Recurrent step
        rnn_input = torch.cat([stochastic, action], dim=-1)
        hidden = self.rnn(rnn_input, hidden)
        
        # Prior distribution
        prior_logits = self.prior(hidden)
        
        # Posterior distribution
        posterior_input = torch.cat([hidden, embed], dim=-1)
        posterior_logits = self.posterior(posterior_input)
        
        # Sample from posterior (during training) or prior (during imagination)
        stochastic = self._sample_categorical(posterior_logits)
        
        return stochastic, hidden, prior_logits, posterior_logits
    
    def _sample_categorical(self, logits):
        """Sample from categorical distribution using Gumbel-Softmax."""
        # Reshape to [batch, stochastic_dim, discrete_dim]
        logits = logits.view(-1, self.stochastic_dim, self.discrete_dim)
        
        # Sample using Gumbel-Softmax
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()
        
        # Flatten back
        return sample.view(-1, self.stochastic_dim * self.discrete_dim)
    
    def imagine(self, action, hidden, stochastic):
        """Imagine next state using prior (no observation)."""
        # Recurrent step
        rnn_input = torch.cat([stochastic, action], dim=-1)
        hidden = self.rnn(rnn_input, hidden)
        
        # Sample from prior
        prior_logits = self.prior(hidden)
        stochastic = self._sample_categorical(prior_logits)
        
        return stochastic, hidden


class RewardPredictor(nn.Module):
    """Predicts rewards from states."""
    
    def __init__(self, state_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return self.net(state)


class ContinuePredictor(nn.Module):
    """Predicts whether episode continues."""
    
    def __init__(self, state_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return torch.sigmoid(self.net(state))


class Actor(nn.Module):
    """Policy network."""
    
    def __init__(self, state_dim=512, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)


class Critic(nn.Module):
    """Value network."""
    
    def __init__(self, state_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return self.net(state)
