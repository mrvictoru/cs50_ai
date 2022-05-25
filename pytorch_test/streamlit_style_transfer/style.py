import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    with torch.no_grad():
        model = TransformerNet()
        state_dict = torch.load(path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

def stylize(style_model, content_image, output_path):
    image = utils.load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = content_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(image).cpu()
    utils.save_image(output_path, output[0])
