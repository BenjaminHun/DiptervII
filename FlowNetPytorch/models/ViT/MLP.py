import torch.nn as nn

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config,i):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"][i], config["intermediate_size"][i])
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(config["intermediate_size"][i], config["hidden_size"][i])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x