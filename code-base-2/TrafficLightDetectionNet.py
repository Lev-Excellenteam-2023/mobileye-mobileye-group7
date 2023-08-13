import torch.nn as nn
from data_utils import MyNeuralNetworkBase
import torch as torch


class TrafficLightDetectionNet(MyNeuralNetworkBase):
    def __init__(self, **kwargs):
        super(TrafficLightDetectionNet, self).__init__(**kwargs)

    def set_net_and_loss(self):
        # Override this method to define your own network architecture and loss function
        self.layers = (
            nn.Conv2d(self.num_in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * (self.w // 4) * (self.h // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.loss_func = nn.BCEWithLogitsLoss
        print()



# Create an instance of your TrafficLightDetectionNet
traffic_light_net = TrafficLightDetectionNet()

# Print the architecture of your network
print(traffic_light_net)
