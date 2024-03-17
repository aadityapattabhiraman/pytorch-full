#!/usr/bin/env python3

import torch
import torchvision.models as models


model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

model = models.vgg16()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()