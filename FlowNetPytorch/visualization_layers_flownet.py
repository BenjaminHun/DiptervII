import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models import FlowNetC
import models
model = FlowNetC.FlowNetC(batchNorm=True)
checkpoint = torch.load("model_best.pth.tar")
device="cpu"
layers = []

network_data = torch.load("model_best.pth.tar")
print("=> using pre-trained model '{}'".format(network_data['arch']))
model = models.__dict__[network_data['arch']](network_data).to(device)
model.eval()
for m in model.modules():
    if type(m) == nn.Conv2d:
        layers.append(m)
print(layers)

img = plt.imread("FlowNetPytorch/00001_img1.jpg")

# plt.imshow(img)
# plt.show()
device = "cpu"
img = torch.from_numpy(img.astype(np.float32)).to(device)
img = img.unsqueeze(0)
img = img.permute(0, 3, 1, 2)

results = [layers[0](img)]
outputs=results
for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer].squeeze()
    print("Layer", num_layer+1)
    for i, f in enumerate(layer_viz):
        plt.subplot(int(layer_viz.shape[0]/8)+1, 8, i+1)
        plt.imshow(f.detach().cpu().numpy())
        plt.axis("off")
    plt.savefig("1.png")
    plt.close()