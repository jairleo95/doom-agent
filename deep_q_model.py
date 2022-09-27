import torch as T
import torch.nn as nn
import torch.nn.functional as F
class DeepQNetwork(nn.Module):
    def __init__(self, device):
        super(DeepQNetwork, self).__init__()
        self.device = device
        #padding "valid" is 0
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # ((84−8+2(0))/4)+1=20
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # ((20−4+2(0))/2)+1=9
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        # ((9−4+2(0))/2)+1=3.5
        self.conv3_bn = nn.BatchNorm2d(128)

        #flatten
        self.flatten = nn.Linear(3*3*128, 512)

        self.fc2 = nn.Linear(512, 3)

        self.loss = nn.MSELoss()


    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        ##observation = observation.view(-1, 1, 84, 84)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1_bn(self.conv1(observation)))
        observation = F.relu(self.conv2_bn(self.conv2(observation)))
        observation = F.relu(self.conv3_bn(self.conv3(observation)))

        observation = observation.view(-1, 3*3*128)
        observation = F.relu(self.flatten(observation))
        actions = self.fc2(observation)
        #print("actions.shape:"+str(actions.shape))
        return actions
