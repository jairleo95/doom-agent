import torch
import torch.nn as nn

from .config import DEVICE


class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)


@torch.no_grad()
def get_value(net, state):
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    _, value = net(state_t)
    return float(value.squeeze(0).item())
