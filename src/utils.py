"""
Modified version from @author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale

def preprocess(image):
    img = rgb_to_grayscale(transforms.Resize((84, 84))(
        torch.from_numpy(image).permute(2, 1, 0)
    ))
    threshold = 1 
    binary_img = (img < threshold).float()
    return binary_img