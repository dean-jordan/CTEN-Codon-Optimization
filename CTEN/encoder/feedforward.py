import torch.nn as nn
from additional import activation

class EncoderFeedForwardNetwork(nn.Module):
    def __init__(self, model, ff):
        super(EncoderFeedForwardNetwork, self).__init__()
        self.recurrent1 = nn.Linear(model, ff)
        self.recurrent2 = nn.Linear(ff, model)
        self.recurrent3 = nn.Linear(ff, model)
        self.recurrent4 = nn.Linear(ff, model)
        self.recurrent5 = nn.Linear(ff, model)
        self.relu = activation.ReLU()

    def forward(self, x):
        return self.recurrent5(
            self.relu(
                self.recurrent4(
                    self.relu(
                        self.recurrent3(
                            self.relu(
                                self.recurrent2(
                                    self.relu(
                                        self.recurrent1(
                                            self.relu(x)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                ))