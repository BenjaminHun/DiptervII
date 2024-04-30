import torch.nn as nn
import torch
from models.ViT.PatchEmbeddings import PatchEmbeddings


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config, i):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config, i)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(
                1, self.patch_embeddings.num_patches, config["hidden_size"][i])).to("cuda")
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):  # 10,3,32,32
        x = self.patch_embeddings(x)  # 10,64,48
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = x + self.position_embeddings  # 10,65,48
        x = self.dropout(x)
        return x
