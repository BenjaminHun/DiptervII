import torch.nn as nn

from models.ViT.Encoder import Encoder
from models.ViT.Embeddings import Embeddings


class ViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.checkpoints = []

        # Create the embedding module

        # Create a linear layer to project the encoder's output to the number of classes
        # Initialize the weights
        # self.apply(self._init_weights)
        self.embedding = nn.ModuleList()
        self.encoder = nn.ModuleList()
        for i in range(self.config["num_depths"]):
            self.embedding.append(Embeddings(self.config, i))
            self.encoder.append(Encoder(self.config, i))

    def forward(self, x):
        for i in range(self.config["num_depths"]):
            # Calculate the embedding output
            x = self.embedding[i](x)
            # Calculate the encoder's output
            x = self.encoder[i](x)  # 10,64,48
            x = x.transpose(-1, -2)
            x = x.view(
                self.config["batch_size"], self.config["hidden_size"][i], self.config["image_size"][i][0]//self.config["patch_size"][0], self.config["image_size"][i][1]//self.config["patch_size"][1])
            self.checkpoints.append(x)

        return self.checkpoints
