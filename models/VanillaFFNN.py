import torch
import torch.nn as nn


class FFNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_normal_(self.fc1.weight)
        print(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.fc2.weight)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        # return torch.round(x)
        return x
