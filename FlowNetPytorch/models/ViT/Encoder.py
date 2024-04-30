import torch.nn as nn

from models.ViT.Block import Block
class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config, i):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config, i)
            self.blocks.append(block)

    def forward(self, x):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x = block(x)
        return x
