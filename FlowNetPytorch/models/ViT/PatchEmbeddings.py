import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config,i):
        super().__init__()
        self.image_size = config["image_size"][i]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"][i]
        self.hidden_size = config["hidden_size"][i]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size[0] // self.patch_size[0]) *(self.image_size[1]//self.patch_size[1])
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size).to("cuda")

    def forward(self, x):  # 10,3,32,32
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)  # 10,48,8,8
        x = x.flatten(2).transpose(1, 2)  # 10,48,64->10.64,48
        return x
