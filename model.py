# Basic neural style transfer implementation using VGG19
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
# Commented out since not using Colab right now
# from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Check if GPU is available, use CPU if not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG model and set to eval mode
vgg = models.vgg19(pretrained=True).features.eval().to(device)

# Layers we want to extract features from
style_layers = [0, 5, 10, 19, 28]  # Various conv layers for style
content_layers = [34]  # Just one layer for content is enough

# Weights for each style layer - equal weights for now
# layers_weights = {0: 0.3, 5: 0.3, 10: 0.3, 19: 0.3, 28: 0.3}
layers_weights = {
    0: 0.25,  # conv1_1
    5: 0.25,  # conv2_1
    10: 0.25, # conv3_1
    19: 0.25, # conv4_1
    28: 0.25  # conv5_1
}

# Calculate content loss
def compute_content_cost(content_feats, generated_feats):
    content_layer = content_layers[0]
    a_C = content_feats[content_layer]
    a_G = generated_feats[content_layer]
    J_content = torch.mean((a_C - a_G) ** 2)
    return J_content

# Helper function for style loss
def gram_matrix(x):
    batch_size, channels, height, width = x.size()
    features = x.view(batch_size, channels, height * width)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(channels * height * width)

# Calculate style loss
def compute_style_cost(style_feats, generated_feats, layers_weights):
    J_style = 0
    for layer_id, weight in layers_weights.items():
        a_S = style_feats[layer_id]
        a_G = generated_feats[layer_id]
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        layer_loss = torch.mean((GS - GG) ** 2)
        J_style += layer_loss * weight
    return J_style

# Training loop will go here