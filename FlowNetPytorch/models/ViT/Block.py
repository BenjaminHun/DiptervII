import torch.nn as nn

from models.ViT.MLP import MLP
from models.ViT.MultiHeadAttention import MultiHeadAttention


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config, i):
        super().__init__()
        self.attention = MultiHeadAttention(config, i)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"][i])
        self.mlp = MLP(config, i)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"][i])

    def forward(self, x):  # 10,65,48
        # Self-attention
        attention_output = \
            self.attention(self.layernorm_1(x))  # 10,65,48
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        return x
