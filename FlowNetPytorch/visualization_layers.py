import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np





model = torchvision.models.alexnet(pretrained=False).features
layers = []

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

#for i in range(1, len(layers)):
#    results.append(layers[i](results[-1]))

outputs = results
print(outputs[0].shape)

for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer].squeeze()
    print("Layer", num_layer+1)
    for i, f in enumerate(layer_viz):
        plt.subplot(int(layer_viz.shape[0]/8)+1, 8, i+1)
        plt.imshow(f.detach().cpu().numpy())
        plt.axis("off")
    plt.savefig("3.png")
    plt.close()



